#include "app.h"

#include <Windows.h>
#include <mmsystem.h>#include "render.h" 
#include <chrono>
#include <stdexcept>
#include <thread>

#pragma comment(lib, "winmm.lib")

/// <summary>
/// Handles the app's lifecycle
/// </summary>
App::~App() {
    m_renderer.waitIdle();   // make sure GPU is done before window dies

    if (m_window) glfwDestroyWindow(m_window);
    glfwTerminate();
}


/// <summary>
/// Initialize the app.
/// </summary>
/// <param name="configPath">Location of config file. Defaults to 'config.json' in .exe location. </param>
/// <returns>True if app successsfully initialized. False otherwise. </returns>
bool App::init(const std::string& configPath) {
    if (!m_config.load(configPath)) return false;
    if (m_config.trials.empty()) {
        Utils::FatalError("[App] No trials in config.");
        return false;
    }

    timeoutDuration = m_config.imageTime;
    flickerRate = m_config.flickerRate;
    waitTimeoutDuration = m_config.waitTime;

    // shuffles the trials and the order of flickers
    Utils::ShuffleTrials(m_config.trials);
    Utils::ShuffleFlickers(m_config.trials);

    // *************** GLFW init and window creation **********************
    if (!glfwInit()) { Utils::FatalError("[App] GLFW init failed"); return false; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    m_monitorWidth = mode->width;
    m_monitorHeight = mode->height;

    m_window = glfwCreateWindow(
        m_monitorWidth * 2, // instead of dealing with two windows, use 1 window stretched to fit 2 monitors. ( this is so i dont have to deal with switching contexts all the time)
        m_monitorHeight, 
        "Flicker Experiment", 
        nullptr, 
        nullptr
    );
    glfwSetWindowUserPointer(m_window, this);
    glfwSetKeyCallback(m_window, keyCallback);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);

    // init the renderer
    if (!m_renderer.init(m_window, m_monitorWidth, m_monitorHeight))
        return false;

    // upload the initial texturess (instructionss, and the first trial textures)
    loadInstructionsTextures();
    loadTexturesForTrial(m_config.trials[0]);

    // initialize the CSV to track responses
    m_csv.init(m_config.participantID, m_config.participantAge, m_config.participantGender 
        ,m_config.conditionName,  { "Index", "Image", "Viewing Mode", "Answer", "Actual", "Reaction Time (s)" }, 
        m_config.outputDirectory.string());

    m_phase = TrialPhase::StartInstructions;
    m_phaseStart = glfwGetTime();
    return true;
}


/// <summary>
/// Main loop, also controls FPS.
/// </summary>
void App::run() {
    using clock = std::chrono::high_resolution_clock;
    const double targetFrameTime = 1.0 / m_config.targetFPS;
    auto nextFrameTime = clock::now();

    while (!glfwWindowShouldClose(m_window) && m_phase != TrialPhase::Done) {
        glfwPollEvents();
        update();

        m_renderer.drawFrame(buildScene()); // draws the scene

        nextFrameTime += std::chrono::duration_cast<clock::duration>( // fps lock
            std::chrono::duration<double>(targetFrameTime));
        std::this_thread::sleep_until(nextFrameTime);
    }
}

/// <summary>
/// Build scene passes on the app's trial state to the renderer.
/// The app handles all timings and trial increments, so for seperation of responsibility, 
/// we need to update a shared 'scene' that the app can update for the renderer to read.
/// </summary>
/// <returns>
/// FrameScene object
/// </returns>
FrameScene App::buildScene() const {
    FrameScene s;
    s.drawCrosshair = true;

    switch (m_phase) {
    case TrialPhase::StartInstructions:
        s.mode = FrameScene::Mode::StartInstructions;
        break;

    case TrialPhase::ShowImages:
        s.mode = FrameScene::Mode::ShowImages;
        s.flickerShow = m_flickerShow;
        s.flickerIndex = (m_trialIndex < (int)m_config.trials.size())
            ? m_config.trials[m_trialIndex].flickerIndex : 0;
        break;

    case TrialPhase::WaitForResponse:
        s.mode = FrameScene::Mode::WaitForResponse;
        break;

    case TrialPhase::Done:
    default:
        s.mode = FrameScene::Mode::Blank;
        break;
    }
    return s;
}

void App::initGame() {
    m_trialIndex = 0;
    m_phase = TrialPhase::ShowImages;
    m_phaseStart = glfwGetTime();
    m_flickerLast = m_phaseStart;
    m_flickerShow = false;
}

/// <summary>
/// The main update loop.
/// </summary>
void App::update() {
    const double now = glfwGetTime();
    const double elapsed = now - m_phaseStart;

    if (m_phase == TrialPhase::ShowImages) {
        if (elapsed >= timeoutDuration) {
            advancePhase();
            return;
        }
        const double flickerInterval = 1.0 / flickerRate;
        if (now - m_flickerLast >= flickerInterval) {
            m_flickerLast = now;
            m_flickerShow = !m_flickerShow;
        }
    }

    pollGamepad(); // collect button press events from the gamepad
}

void App::advancePhase() {
    m_phase = TrialPhase::WaitForResponse;
    m_phaseStart = glfwGetTime();
    m_responseStart = m_phaseStart;

    if ((m_trialIndex + 1) < (int)m_config.trials.size())
        loadTexturesForTrial(m_config.trials[m_trialIndex + 1]);
}
/// <summary>
/// Records user's response in CSV file
/// </summary>
/// <param name="key"></param>
void App::recordResponse(int key) {
    if (m_phase != TrialPhase::ShowImages && m_phase != TrialPhase::WaitForResponse)
        return;

    TrialResult result;
    result.imageName = m_config.trials[m_trialIndex].name;
    result.answer = (key == GLFW_KEY_LEFT) ? 0 : 1;
    result.actual = m_config.trials[m_trialIndex].flickerIndex;
    result.index = m_trialIndex;

    // play sound based on if response is correct or incorrect
    result.answer == result.actual ? PlaySound(TEXT("./assets/sounds/Success.wav"), NULL, SND_FILENAME | SND_ASYNC) : PlaySound(TEXT("./assets/sounds/error.wav"), NULL, SND_FILENAME | SND_ASYNC);

    // translate viewing mode into name data
    switch (m_config.trials[m_trialIndex].viewingMode) {
        case 0:  result.viewingMode = "Stereo"; break;
        case 1:  result.viewingMode = "Left";   break;
        case 2:  result.viewingMode = "Right";  break;
        default: result.viewingMode = "N/A";    break;
    }

    // do we need reaction time???
    result.reactionTime = glfwGetTime() - m_responseStart;
    m_results.push_back(result);

    m_csv.writeRow({
        std::to_string(result.index),
        result.imageName,
        result.viewingMode,
        std::to_string(result.answer),
        std::to_string(result.actual),
        std::to_string(result.reactionTime)
    });

    m_trialIndex++;

    if (m_trialIndex >= (int)m_config.trials.size()) {
        m_phase = TrialPhase::Done;
        return;
    }

    // load the textures for the upcoming trial
    if (m_phase == TrialPhase::ShowImages) loadTexturesForTrial(m_config.trials[m_trialIndex]);

    m_phase = TrialPhase::ShowImages;
    m_phaseStart = glfwGetTime();
    m_responseStart = m_phaseStart;
    m_flickerShow = false;
    m_flickerLast = m_phaseStart;
}

/// <summary>
/// Polls the gamepad for button presses
/// </summary>
void App::pollGamepad() {
    GLFWgamepadstate state;
    if (!glfwGetGamepadState(GLFW_JOYSTICK_1, &state)) return;

    const bool aPressed = state.buttons[GLFW_GAMEPAD_BUTTON_A];
    const bool leftPressed = state.buttons[GLFW_GAMEPAD_BUTTON_X];
    const bool rightPressed = state.buttons[GLFW_GAMEPAD_BUTTON_B];

    if (aPressed && !m_prevGamepadA && m_phase == TrialPhase::StartInstructions)
        initGame();

    if (leftPressed && !m_prevGamepadLeft)  recordResponse(GLFW_KEY_LEFT);
    if (rightPressed && !m_prevGamepadRight) recordResponse(GLFW_KEY_RIGHT);

    m_prevGamepadA = aPressed;
    m_prevGamepadLeft = leftPressed;
    m_prevGamepadRight = rightPressed;
}

// loading textures


// hardcoded to load textures for instructions. these are loaded in once per program's lifecycle

/// <summary>
/// Loads the instructions textures (starting screen, waiting for response). 
/// These are loaded once per program lifecycle, and are kept as unchanging textures throughout.
/// </summary>
void App::loadInstructionsTextures() {
    m_renderer.uploadTexture(TEX_WAIT_L, "./assets/instructions/responsescreen_L.ppm");
    m_renderer.uploadTexture(TEX_START_L, "./assets/instructions/startscreen_L.ppm");
    m_renderer.uploadTexture(TEX_WAIT_R, "./assets/instructions/responsescreen_R.ppm");
    m_renderer.uploadTexture(TEX_START_R, "./assets/instructions/startscreen_R.ppm");
}

/// <summary>
/// // load the textures for the current trial based on the viewing mode. these are loaded every time the trial is switched.
/// </summary>
/// <param name="img">The paths to the 4 image permuatationss (L, R, Original, Degraded) </param>
void App::loadTexturesForTrial(const ImagePaths& img) {
    switch (img.viewingMode) {
    case 0: // stereo
        m_renderer.uploadTexture(TEX_ORIG_L, img.L_orig.string());
        m_renderer.uploadTexture(TEX_ORIG_R, img.R_orig.string());
        m_renderer.uploadTexture(TEX_DEC_L, img.L_dec.string());
        m_renderer.uploadTexture(TEX_DEC_R, img.R_dec.string());
        break;
    case 1: // left only
        m_renderer.uploadTexture(TEX_ORIG_L, img.L_orig.string());
        m_renderer.uploadTexture(TEX_ORIG_R, img.L_orig.string());
        m_renderer.uploadTexture(TEX_DEC_L, img.L_dec.string());
        m_renderer.uploadTexture(TEX_DEC_R, img.L_dec.string());
        break;
    case 2: // right only
        m_renderer.uploadTexture(TEX_ORIG_L, img.R_orig.string());
        m_renderer.uploadTexture(TEX_ORIG_R, img.R_orig.string());
        m_renderer.uploadTexture(TEX_DEC_L, img.R_dec.string());
        m_renderer.uploadTexture(TEX_DEC_R, img.R_dec.string());
        break;
    default:
        Utils::FatalError("[App] Invalid viewing mode: " + std::to_string(img.viewingMode) + ". Must be one of: 0 (stereo), 1 (left only), 2 (right only) ");
    }
}

// GLFW callbacks

void App::keyCallback(GLFWwindow* window, int key, int /*scancode*/,
    int action, int /*mods*/)
{
    if (action != GLFW_PRESS) return;
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, true);
        return;
    }
    if (key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT)
        app->recordResponse(key);

    if (key == GLFW_KEY_ENTER && app->m_phase == TrialPhase::StartInstructions)
        app->initGame();
}