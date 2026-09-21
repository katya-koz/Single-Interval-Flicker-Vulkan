#include "app.h"

#include <Windows.h>
#include <mmsystem.h>
#include "render.h" 
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <thread>
#pragma comment(lib, "winmm.lib")



/// <summary>
/// Handles the app's lifecycle
/// </summary>
App::~App() {
    if (m_decodeThread.joinable())
        m_decodeThread.join();
    m_renderer.waitIdle();   // make sure GPU is done before window dies

    if (m_window) glfwDestroyWindow(m_window);
    glfwTerminate();
}


/// <summary>
/// Initialize the app.
/// </summary>
/// <param name="configPath">Location of config file. Defaults to 'config.json' in .exe location. </param>
/// <returns>True if app successsfully initialized. False otherwise. </returns>
bool App::init(const std::string& configPath, std::string& inputPath) {
    if (!m_config.loadConfig(configPath)) return false;
    if (!m_config.loadTrials(inputPath)) return false;


    if (m_config.trials.empty()) {
        Utils::FatalError("[App] No trials found in config.");
        return false;
    }

    timeoutDuration = m_config.imageTime;
    flickerRate = m_config.flickerRate;
    m_flickerInterval = 1.0 / (flickerRate * 2.0);

    waitTimeoutDuration = m_config.waitTime;

    // shuffles the trials and the order of flickers
    // commented out: trial order determined by trials.csv
    //Utils::ShuffleTrials(m_config.trials);
    //Utils::ShuffleFlickers(m_config.trials);

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

    #ifndef DEBUG_MOUSE_GAZE
        glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // disable cursor
    #endif
    // init the renderer
    if (!m_renderer.init(m_window, m_monitorWidth, m_monitorHeight, m_config.displayMode))
        return false;

    // upload the initial texturess (instructionss, and the first trial textures)
    loadInstructionsTextures();
    loadTexturesForTrial(m_config.trials[0]);

    // initialize the CSV to track responses
    m_csv.init(
        m_config.experimentInfo.participantID,
        m_config.experimentInfo.participantAge,
        m_config.experimentInfo.participantGender,
        m_config.experimentInfo.block,
        m_config.experimentInfo.sessionNumber,
        m_config.experimentInfo.groupNumber,
        //m_config.intervalMode,  dont need for now
        //m_config.displayMode, dont need for now
            { 
                "Codec",
                "Foveation",
                "Image", 
                "Interval", 
                "Position-X (Left)", 
                "Position-Y (Left)",
                "Position-X (Right)",
                "Position-Y (Right)",
                "Mode",
                "Response",
                "Duration",
                "Subject",
                "Mean Gaze X (Left)",
                "Mean Gaze Y (Left)",
                "Mean Gaze X (Right)",
                "Mean Gaze Y (Right)",
                "Std. Dev. Gaze X (Left)",
                "Std. Dev. Gaze Y (Left)",
                "Std. Dev. Gaze X (Right)",
                "Std. Dev. Gaze Y (Right)",
            },
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
    //const double targetFrameTime = 1.0 / m_config.targetFPS; // removed. messes with timing state switches
    auto nextFrameTime = clock::now();

    while (!glfwWindowShouldClose(m_window) && m_phase != TrialPhase::Done) {
        glfwPollEvents();
        update();
        #ifdef DEBUG_MOUSE_GAZE
             m_eyetracker.updateMouseGaze(m_window, m_monitorWidth);
        #endif
        m_renderer.drawFrame(buildScene()); // draws the scene
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
    //s.drawFixationPoint = true;

    s.fixationPointCoords = m_config.trials[m_trialIndex].fixationCoords;

    switch (m_phase) {
    case TrialPhase::StartInstructions:
    {
        s.mode = FrameScene::Mode::StartInstructions;
        break;
    }

    case TrialPhase::ShowSideBySideImages:
    {
        s.mode = FrameScene::Mode::ShowSingleIntervalImages;
        s.flickerShow = m_flickerShow;
        s.flickerIndex = (m_trialIndex < (int)m_config.trials.size()) ? m_config.trials[m_trialIndex].flickerIndex : 0;
        break;
    }

    case TrialPhase::ShowFullFieldImage: {
        const int flickerIndex = (m_trialIndex < (int)m_config.trials.size()) ? m_config.trials[m_trialIndex].flickerIndex : 0;
        const bool isFlickerInterval = (flickerIndex == m_interTrialImageIndex);

        s.mode = isFlickerInterval ? FrameScene::Mode::ShowFlickerImage : FrameScene::Mode::ShowImage;
        s.flickerShow = m_flickerShow;
        break;
    }
    case TrialPhase::ShowBuffer:
    {
        s.mode = FrameScene::Mode::ShowBuffer;
        break;
    }

    case TrialPhase::WaitForResponse:
    {
        s.mode = FrameScene::Mode::WaitForResponse;
        break;
    }

    case TrialPhase::Done:
    {}
    default:
    {
        s.mode = FrameScene::Mode::Blank;
        break;
    }
    }
    return s;
}

void App::initGame() {
    m_trialIndex = 0;
    m_interTrialImageIndex = 0;

    m_phase = TrialPhase::ShowBuffer;
    //if(m_config.intervalMode == 1 ){
    //    m_phase = TrialPhase::ShowSideBySideImages;
    //}
    //else {
    //    m_phase = TrialPhase::ShowFullFieldImage;
    //}
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

    if (m_phase == TrialPhase::ShowSideBySideImages) {
        if (elapsed >= timeoutDuration) {
            advancePhase();
            return;
        }
        
        if (now - m_flickerLast >= m_flickerInterval) {
            m_flickerLast = now;
            m_flickerShow = !m_flickerShow;
        }
    }


    else if (m_phase == TrialPhase::ShowFullFieldImage) {
        if (elapsed >= timeoutDuration) {
            m_eyetracker.stopCollection();
            //std::cout << "elapsed time: " << elapsed << " seconds \n";
            if (m_interTrialImageIndex == 0) {
                // if it's the first image, show the blank buffer screen next
                m_interTrialImageIndex++;
                showBuffer();
                
            }
            else {
                // second image, advance trial and collect response
                m_interTrialImageIndex = 0;
                advancePhase();
            }
            return;
            
        }
        if(m_config.trials[m_trialIndex].flickerIndex == m_interTrialImageIndex) { // flicker only if this is the flicker index
            if (now - m_flickerLast >= m_flickerInterval) {
                //std::cout << "flicker time: " << (now - m_flickerLast) << " seconds \n";
                m_flickerLast = now;
                m_flickerShow = !m_flickerShow;
            }
        }
        else {
            m_flickerShow = false;
        }
    }

    else if (m_phase == TrialPhase::ShowBuffer) {
        if (elapsed >= waitTimeoutDuration) {
            showNextImageInTrial();
        }
    }

    pollGamepad(); // collect button press events from the gamepad
}

void App::advancePhase() {
    m_phase = TrialPhase::WaitForResponse;
    m_phaseStart = glfwGetTime();
    m_responseStart = m_phaseStart;

    // while waiting for response, begin loading/processing images on a different thread 
    if (m_decodeThread.joinable())
        m_decodeThread.join();

    if ((m_trialIndex + 1) < (int)m_config.trials.size()) {
        m_decodeThread = std::thread(&App::decodeImagesForTrial, this, m_config.trials[m_trialIndex + 1]);
    }
}


void App::showBuffer() {
    m_phase = TrialPhase::ShowBuffer;
    m_eyetracker.startCollection();
    m_phaseStart = glfwGetTime();
}

//void App::showNextImageInTrial() {
//    m_phase = TrialPhase::ShowFullFieldImage;
//    m_phaseStart = glfwGetTime();
//}
// added buffer between single interval mode early exit
void App::showNextImageInTrial() {
    if (m_config.intervalMode == 1) {
        // single interval mode
        m_phase = TrialPhase::ShowSideBySideImages;
        m_phaseStart = glfwGetTime();
        m_responseStart = m_phaseStart;
        m_flickerShow = false;
        m_flickerLast = m_phaseStart;
    }
    else {
        // 2 interval mode
        m_phase = TrialPhase::ShowFullFieldImage;
        m_phaseStart = glfwGetTime();
        m_flickerShow = false;
        m_flickerLast = m_phaseStart;
    }
}




/// <summary>
/// Records user's response in CSV file
/// </summary>
/// <param name="key"></param>
void App::recordResponse(int key) {
    // only record response if currently waiting for response, or doing side by side image view
    if (m_phase != TrialPhase::ShowSideBySideImages && m_phase != TrialPhase::WaitForResponse)
        return;

    TrialResult result;

    // not sure if reaction time is needed
    if (m_config.intervalMode == 0) { // 2 interval mode - start counting reaction tiome from response start
        result.reactionTimeMS = (glfwGetTime() - m_responseStart) * 1000; // get time in ms

    }
    else { // single interval mode - start counting reaction time from image shown
        result.reactionTimeMS = (glfwGetTime() - m_phaseStart) * 1000; // get time in ms
    }

    result.imageName = m_config.trials[m_trialIndex].name;
    result.codec = m_config.trials[m_trialIndex].codec;
    result.positionX_L = m_config.trials[m_trialIndex].fixationCoords.Left.X;
    result.positionY_L = m_config.trials[m_trialIndex].fixationCoords.Left.Y;
    result.positionX_R = m_config.trials[m_trialIndex].fixationCoords.Right.X;
    result.positionY_R = m_config.trials[m_trialIndex].fixationCoords.Right.Y;
    result.foveatLevel = m_config.trials[m_trialIndex].foveatLevel;

    // translate viewing mode into name data
    switch (m_config.trials[m_trialIndex].viewingMode) {
    case 0:  result.viewingMode = "Stereo"; break;
    case 1:  result.viewingMode = "Mono Left";   break;
    case 2:  result.viewingMode = "Mono Right";  break;
    default: result.viewingMode = "N/A";    break;
    }

    // if this is in single interval mode, need to flip the answer since it is left/right
    // in two interval mode, the answer is based on the first vs second image, so no 
    // need to flip the answer value in that case.
    if (m_config.intervalMode == 1) {
        // remember that the mirrors flip the image, so the left key actually refers to the right image and vice versa.
        result.response = (key == GLFW_KEY_LEFT) ? 1 : 0;
    }
    else {
        result.response = (key == GLFW_KEY_LEFT) ? 0 : 1;
    }
    
    
    result.actual = m_config.trials[m_trialIndex].flickerIndex;

    // play sound based on if response is correct or incorrect
    //result.response == result.actual ? PlaySound(TEXT("./assets/sounds/Success.wav"), NULL, SND_FILENAME | SND_ASYNC) : PlaySound(TEXT("./assets/sounds/error.wav"), NULL, SND_FILENAME | SND_ASYNC);

    if (result.response != result.actual) { // only play on incorrect
        const auto soundPath = Utils::getExecutableDirectory() / "assets" / "sounds" / "error.wav";

        PlaySound( soundPath.c_str(), NULL, SND_FILENAME | SND_ASYNC );
    }
        
    m_results.push_back(result);

    // calculate eye tracking statistics


    Utils::GazeStatistics gazeStats = Utils::calculateGazeStatistics(
        m_eyetracker.getSamples(),
        m_monitorWidth, m_monitorHeight,
        m_config.trials[m_trialIndex].imageWidth, m_config.trials[m_trialIndex].imageHeight,   // real stimulus resolution
        result.positionX_L, result.positionY_L,
        result.positionX_R, result.positionY_R
    );

    m_eyetracker.clearSamples();
    m_csv.writeRow(
        {
            result.codec,
            result.foveatLevel,
            result.imageName,
            std::to_string(result.actual),
            std::to_string(result.positionX_L),
            std::to_string(result.positionY_L),
            std::to_string(result.positionX_R),
            std::to_string(result.positionY_R),
            result.viewingMode,
            std::to_string(result.response),
            std::to_string((int)(result.reactionTimeMS)),
            m_config.experimentInfo.participantID,
            std::to_string(gazeStats.leftMean_X),
            std::to_string(gazeStats.leftMean_Y),
            std::to_string(gazeStats.rightMean_X),
            std::to_string(gazeStats.rightMean_Y),
            std::to_string(gazeStats.leftStdDev_X),
            std::to_string(gazeStats.leftStdDev_Y),
            std::to_string(gazeStats.rightStdDev_X),
            std::to_string(gazeStats.rightStdDev_Y)
        });

    if (m_decodeThread.joinable()) {
        m_decodeThread.join();
        uploadDecodedTextures(); // after joining, all images are decoded. so we can upload them to the gpu now

    }
        

    m_trialIndex++;

    if (m_trialIndex >= (int)m_config.trials.size()) {
        m_phase = TrialPhase::Done;
        return;
    }

    // load the textures for the upcoming trial (if the response was recorded 
    // before the waitforresponse screen)
    if (m_phase == TrialPhase::ShowSideBySideImages)
    {
        loadTexturesForTrial(m_config.trials[m_trialIndex]);
    }

    if (m_config.intervalMode == 1) { // single interval mode 
        //m_phase = TrialPhase::ShowSideBySideImages;
        showBuffer();
    }
    else { // 2 interval mode
        //m_phase = TrialPhase::ShowFullFieldImage;
        showBuffer();
    }

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
    const bool leftPressed =  state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER] > 0.5f;

    const bool rightPressed = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] > 0.5f;

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
void App::loadInstructionsTextures()
{
    const auto assetsDir = Utils::getExecutableDirectory() / "assets" / "instructions";

    if (m_config.intervalMode == 0) { // 2 interval mode
        m_renderer.decodeAndUploadTexture(TEX_WAIT_L, (assetsDir / "responsescreen0_L.ppm").string());

        m_renderer.decodeAndUploadTexture(TEX_WAIT_R,(assetsDir / "responsescreen0_R.ppm").string());
    }
    else { // single interval mode
        m_renderer.decodeAndUploadTexture(TEX_WAIT_L,(assetsDir / "responsescreen1_L.ppm").string());

        m_renderer.decodeAndUploadTexture(TEX_WAIT_R, (assetsDir / "responsescreen1_R.ppm").string());
    }

    m_renderer.decodeAndUploadTexture(TEX_START_L, (assetsDir / "startscreen_L.ppm").string());

    m_renderer.decodeAndUploadTexture(TEX_START_R,(assetsDir / "startscreen_R.ppm").string());
}


/// <summary>
/// // load the textures for the current trial based on the viewing mode. these are loaded every time the trial is switched.
/// </summary>
/// <param name="img">The paths to the 4 image permuatationss (L, R, Original, Degraded) </param>
void App::loadTexturesForTrial(const ImagePaths& img) {
    switch (img.viewingMode) {
    case 0: // stereo
        m_renderer.decodeAndUploadTexture(TEX_ORIG_L, img.L_orig.string());
        m_renderer.decodeAndUploadTexture(TEX_ORIG_R, img.R_orig.string());
        m_renderer.decodeAndUploadTexture(TEX_DEC_L, img.L_dec.string());
        m_renderer.decodeAndUploadTexture(TEX_DEC_R, img.R_dec.string());
        break;
    case 1: // left only
        m_renderer.decodeAndUploadTexture(TEX_ORIG_L, img.L_orig.string());
        m_renderer.decodeAndUploadTexture(TEX_ORIG_R, img.L_orig.string());
        m_renderer.decodeAndUploadTexture(TEX_DEC_L, img.L_dec.string());
        m_renderer.decodeAndUploadTexture(TEX_DEC_R, img.L_dec.string());
        break;
    case 2: // right only
        m_renderer.decodeAndUploadTexture(TEX_ORIG_L, img.R_orig.string());
        m_renderer.decodeAndUploadTexture(TEX_ORIG_R, img.R_orig.string());
        m_renderer.decodeAndUploadTexture(TEX_DEC_L, img.R_dec.string());
        m_renderer.decodeAndUploadTexture(TEX_DEC_R, img.R_dec.string());
        break;
    default:
        Utils::FatalError("[App] Invalid viewing mode: " + std::to_string(img.viewingMode) + ". Must be one of: 0 (stereo), 1 (left only), 2 (right only) ");
    }
}
/// <summary>
/// This is to seperately decode images (which is cpu intensive)
/// </summary>
/// <param name="img"></param>
void App::decodeImagesForTrial(const ImagePaths& img) {
    switch (img.viewingMode) {
    case 0: // stereo
        decodeImageForUpload(TEX_ORIG_L, img.L_orig.string());
        decodeImageForUpload(TEX_ORIG_R, img.R_orig.string());
        decodeImageForUpload(TEX_DEC_L, img.L_dec.string());
        decodeImageForUpload(TEX_DEC_R, img.R_dec.string());
        break;
    case 1: // left only
        decodeImageForUpload(TEX_ORIG_L, img.L_orig.string());
        decodeImageForUpload(TEX_ORIG_R, img.L_orig.string());
        decodeImageForUpload(TEX_DEC_L, img.L_dec.string());
        decodeImageForUpload(TEX_DEC_R, img.L_dec.string());
        break;
    case 2: // right only
        decodeImageForUpload(TEX_ORIG_L, img.R_orig.string());
        decodeImageForUpload(TEX_ORIG_R, img.R_orig.string());
        decodeImageForUpload(TEX_DEC_L, img.R_dec.string());
        decodeImageForUpload(TEX_DEC_R, img.R_dec.string());
        break;
    default:
        Utils::FatalError("[App] Invalid viewing mode: " + std::to_string(img.viewingMode) + ". Must be one of: 0 (stereo), 1 (left only), 2 (right only) ");
    }
}

/// <summary>
/// Helper to decode 1 single image. Wrapper for renderer method.
/// Saves decoded image to memory on complete.
/// </summary>
/// <param name="path"></param>

void App::decodeImageForUpload(TextureSlot slot, const std::string& path) {
    DecodedImage img = m_renderer.decodeImageForUpload(slot, path);
}

void App::uploadDecodedTexture(TextureSlot slot) {
    m_renderer.uploadDecodedTexture(slot);
}

void App::uploadDecodedTextures() {
    uploadDecodedTexture(TEX_ORIG_L);
    uploadDecodedTexture(TEX_ORIG_R);
    uploadDecodedTexture(TEX_DEC_L);
    uploadDecodedTexture(TEX_DEC_R);
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