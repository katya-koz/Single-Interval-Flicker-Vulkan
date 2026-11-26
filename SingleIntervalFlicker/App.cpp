#include "app.h"
#include <cmath>
#include <Windows.h>
#include <mmsystem.h>
#include "render.h" 
#include "tobii_research.h"
#include <algorithm>
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
    m_eyetracker.stop(m_tobiiHandle); // unsub from eye tracker
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


    m_displayInfo.viewingDistanceCm = m_config.experimentInfo.viewingDistanceCm;

    timeoutDuration = m_config.experimentInfo.imageTime;
    flickerRate = m_config.experimentInfo.flickerRate;
    m_flickerInterval = 1.0 / (flickerRate * 2.0);


    m_maxFixationMs = static_cast<int>(m_config.experimentInfo.maxFixationTimeS * 1000);
    // max fixation is enabled only when eye tracking is enabled and the max fixation time is greater than the fixation time ms
    isMaxFixation = m_config.experimentInfo.isEyeTracking && m_maxFixationMs > m_config.experimentInfo.fixationTimeMs;
    waitTimeoutDuration = m_config.experimentInfo.waitTime;

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
    m_displayInfo.pixelWidth = mode->width;
    m_displayInfo.pixelHeight = mode->height;


    m_window = glfwCreateWindow(
        m_displayInfo.pixelWidth * 2, // instead of dealing with two windows, use 1 window stretched to fit 2 monitors. ( this is so i dont have to deal with switching contexts all the time)
        m_displayInfo.pixelHeight,
        "Flicker Experiment",
        nullptr,
        nullptr
    );
    glfwSetWindowUserPointer(m_window, this);
    glfwSetKeyCallback(m_window, keyCallback);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    glfwGetMonitorPhysicalSize(monitor, &m_displayInfo.physicalWidthMm, &m_displayInfo.physicalHeightMm); // returns sizes in mm

    // we need to load trials here since we have to init the display info before loading
    if (!m_config.loadTrials(inputPath, m_displayInfo)) return false;

    if (m_config.trials.empty()) {
        Utils::FatalError("[App] No trials found in config.");
        return false;
    }

    std::cerr << "[Config] Successfully loaded: " + configPath + "\n";
    std::cout << m_config.toString() << std::endl;


    if (m_config.experimentInfo.isEyeTracking) {
        m_eyetracker.init(
            m_config.experimentInfo,
            m_displayInfo,
            m_config.blockInfo
        );
    }

    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // disable cursor

    // init the renderer
    if (!m_renderer.init(m_window, m_displayInfo.pixelWidth, m_displayInfo.pixelHeight, m_config.experimentInfo.displayMode))
        return false;

    // upload the initial texturess (instructionss, and the first trial textures)
    loadInstructionsTextures();
    loadTexturesForTrial(m_config.trials[0]);

    // initialize the CSV to track responses
    m_csv.init(
        m_config.blockInfo,
        {
            "Trial Index",
            "Image ID",
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
            "Fixation Window (ms)",
            "Max Mean Position (pixels)",
            "Max Spread (pixels)",

            "Fixation 1 Mean Position (pixels)",
            "Fixation 2 Mean Position (pixels)",

            "Fixation 1 Spread L (pixels)",
            "Fixation 1 Spread R (pixels)",

            "Fixation 2 Spread L (pixels)",
            "Fixation 2 Spread R (pixels)"
        },
        true, // has metadata header
        ""); // no tag

    // *************** find and connect to the eye tracker **********************
    if (m_config.experimentInfo.isEyeTracking) {
        TobiiResearchEyeTrackers* eyetrackers = nullptr;
        TobiiResearchStatus findStatus = tobii_research_find_all_eyetrackers(&eyetrackers);

        if (findStatus != TOBII_RESEARCH_STATUS_OK || !eyetrackers || eyetrackers->count == 0) {
            Utils::FatalError("[App] No Tobii eye tracker found.");
            return false;
        }

        m_tobiiHandle = eyetrackers->eyetrackers[0]; // handle stays valid after the list itself is freed
        tobii_research_free_eyetrackers(eyetrackers);

        m_eyetracker.start(m_tobiiHandle); // subscribe to gaze data

        std::cerr << "Eye tracking is set to TRUE for this block.\n";
    }
    else {
        std::cerr << "[WARNING] Eye tracking is set to FALSE for this block. Gaze data will not be recorded; 'Wait Time (s)' is the fixation time; Gaze statistics in output csv will be represented with '-1'.\n";
    }

    m_phase = TrialPhase::StartInstructions;

    if (m_config.experimentInfo.isEyeTracking) {
        m_eyetracker.updateStage(TrialPhase::StartInstructions, -1);
        m_eyetracker.updateEventNumber(-1); // start instructions are not a trial and therefore don't have an applicable event number
    }

    m_phaseStart = glfwGetTime();
    return true;
}

/// <summary>
/// Main loop, also controls FPS.
/// </summary>
void App::run() {
    if (m_config.experimentInfo.isEyeTracking) {
        m_eyetracker.markExperimentStart(); // marks when to record time stamps from tobii tracker
    }
    while (!glfwWindowShouldClose(m_window) && m_phase != TrialPhase::Done) {
        glfwPollEvents();
        update();
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

    // m_trialIndex is -1 once the experiment is Done, so guard the lookup
    if (m_phase != TrialPhase::StartInstructions &&
        m_trialIndex >= 0 && m_trialIndex < (int)m_config.trials.size())
    {
        s.fixationPointCoords = m_config.trials[m_trialIndex].fixationCoords;
    }

    if (m_config.experimentInfo.isEyeTracking) {

        TobiiResearchGazeData g;
        if (m_eyetracker.getLatestSample(g)) {
            s.leftGazeNorm = {
                g.left_eye.gaze_point.position_on_display_area.x,
                g.left_eye.gaze_point.position_on_display_area.y,
                g.left_eye.gaze_point.validity == TOBII_RESEARCH_VALIDITY_VALID
            };
            s.rightGazeNorm = {
                g.right_eye.gaze_point.position_on_display_area.x,
                g.right_eye.gaze_point.position_on_display_area.y,
                g.right_eye.gaze_point.validity == TOBII_RESEARCH_VALIDITY_VALID
            };
        }
    }

    switch (m_phase) {
    case TrialPhase::StartInstructions:
    {
        s.mode = TrialPhase::StartInstructions;
        break;
    }

    case TrialPhase::ShowSideBySideImages:
    {
        s.mode = TrialPhase::ShowSideBySideImages;
        s.flickerShow = m_flickerShow;
        s.flickerIndex = (m_trialIndex >= 0 && m_trialIndex < (int)m_config.trials.size()) ? m_config.trials[m_trialIndex].flickerIndex : 0;
        break;
    }

    case TrialPhase::ShowImage: {
        const int flickerIndex = (m_trialIndex >= 0 && m_trialIndex < (int)m_config.trials.size()) ? m_config.trials[m_trialIndex].flickerIndex : 0;
        const bool isFlickerInterval = (flickerIndex == m_interTrialImageIndex);

        s.mode = isFlickerInterval ? TrialPhase::ShowFlickerImage : TrialPhase::ShowImage;
        s.flickerShow = m_flickerShow;
        break;
    }


    case TrialPhase::ShowBuffer:
    {
        s.mode = TrialPhase::ShowBuffer;
        break;
    }

    case TrialPhase::WaitForResponse:
    {
        s.mode = TrialPhase::WaitForResponse;
        break;
    }

    case TrialPhase::Done:
    {}
    default:
    {
        s.mode = TrialPhase::Blank;
        break;
    }
    }
    return s;
}

void App::initGame() {

    m_trialIndex = 0;
    m_interTrialImageIndex = 0;

    if (m_config.experimentInfo.isEyeTracking) {
        m_eyetracker.clearSamples();
    }
    m_phase = TrialPhase::ShowBuffer;
    updateEyeTrackerStage();
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

    if (m_phase == TrialPhase::ShowImage) {
        if (elapsed >= timeoutDuration) {
            m_eyetracker.stopCollection();
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
        if (m_config.trials[m_trialIndex].flickerIndex == m_interTrialImageIndex) { // flicker only if this is the flicker index
            if (now - m_flickerLast >= m_flickerInterval) {
                m_flickerLast = now;
                m_flickerShow = !m_flickerShow;
            }
        }
        else {
            m_flickerShow = false;
        }
    }

    else if (m_phase == TrialPhase::ShowBuffer) {

        // Gaze is checked first so the fixation stats always hold the freshest window
        // before we either proceed or skip (a skipped trial records the last window).
        bool gazeWithinError = true;

        if (m_config.experimentInfo.isEyeTracking) {
            gazeWithinError = m_eyetracker.gazeWithinError(
                m_config.trials[m_trialIndex].maxMeanPositionPixels,
                m_config.trials[m_trialIndex].maxSpreadPixels,
                m_config.trials[m_trialIndex].fixationCoords,
                &m_config.trials[m_trialIndex].fixationStats,
                m_interTrialImageIndex   // event number: 0 = first fixation, 1 = second
            );
        }

        if ((m_config.experimentInfo.isEyeTracking && bypassGaze) ||
            (elapsed >= waitTimeoutDuration && gazeWithinError)) {
            // fixation achieved (or manually bypassed): show the stimulus
            bypassGaze = false;
            showNextImageInTrial();
        }
        else if (isMaxFixation && elapsed >= m_config.experimentInfo.maxFixationTimeS) {
            // max fixation time exceeded in either the 1st or 2nd fixation:
            // append a copy of this trial for retrial, record this one, and move on.
            bypassGaze = false;

            TrialInformation retry = m_config.trials[m_trialIndex];
            retry.fixationStats = FixationStats(); // retrial must not inherit the failed attempt's stats
            m_config.trials.push_back(std::move(retry));

            skipTrial();
        }
    }
    pollGamepad(); // collect button press events from the gamepad
}

void App::advancePhase() {
    m_phase = TrialPhase::WaitForResponse;
    updateEyeTrackerStage();
    m_phaseStart = glfwGetTime();
    m_responseStart = m_phaseStart;

    // while waiting for response, begin loading/processing images on a different thread 
    if (m_decodeThread.joinable())
        m_decodeThread.join();

    if ((m_trialIndex + 1) < (int)m_config.trials.size()) {
        m_decodeThread = std::thread(&App::decodeImagesForTrial, this, m_config.trials[m_trialIndex + 1]);
    }
}
void App::updateEyeTrackerStage() {

    if (m_config.experimentInfo.isEyeTracking) {
        m_eyetracker.updateStage(m_phase, m_trialIndex);
    }
}
//
//void App::clearEyeTrackingSamples() {
//    if (m_config.experimentInfo.isEyeTracking) {
//        m_eyetracker.updateStage(m_phase, m_trialIndex);
//    }
//
//};
//void App::setEyeTrackingTimeStart() {
//    if (m_config.experimentInfo.isEyeTracking) {
//        m_eyetracker.updateStage(m_phase, m_trialIndex);
//    }
//}

/// <summary>
/// Update the stage every time the phase changes (fixation, stim, start, done, wait for response... )
/// </summary>
void App::updateEyeTrackerStage() {

    if (m_config.experimentInfo.isEyeTracking) {
        m_eyetracker.updateStage(m_phase, m_trialIndex);
        if (m_phase != TrialPhase::ShowBuffer && m_phase != TrialPhase::ShowFlickerImage && m_phase != TrialPhase::ShowImage) {
            // any other phase is not part of a trial, and should therefore not have an event number attached to it
            m_eyetracker.updateEventNumber(-1);
        }
        else {
            m_eyetracker.updateEventNumber(m_interTrialImageIndex);
        }
    }
}


void App::showBuffer() {
    m_phase = TrialPhase::ShowBuffer;
    updateEyeTrackerStage();

    m_eyetracker.startCollection();
    m_phaseStart = glfwGetTime();
}

void App::showNextImageInTrial() {
    m_phase = TrialPhase::ShowImage;
    updateEyeTrackerStage();
    m_phaseStart = glfwGetTime();
    m_flickerShow = false;
    m_flickerLast = m_phaseStart;
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

    if (m_config.experimentInfo.intervalMode == 0) { // 2 interval mode - start counting reaction time from response start
        result.reactionTimeMS = (glfwGetTime() - m_responseStart) * 1000; // get time in ms
    }
    else { // single interval mode - start counting reaction time from image shown
        result.reactionTimeMS = (glfwGetTime() - m_phaseStart) * 1000; // get time in ms
    }

    result.imageName = m_config.trials[m_trialIndex].name;
    result.imageIndex = m_config.trials[m_trialIndex].index;
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
    if (m_config.experimentInfo.intervalMode == 1) {
        // remember that the mirrors flip the image, so the left key actually refers to the right image and vice versa.
        result.response = (key == GLFW_KEY_LEFT) ? 1 : 0;
    }
    else {
        result.response = (key == GLFW_KEY_LEFT) ? 0 : 1;
    }

    result.actual = m_config.trials[m_trialIndex].flickerIndex;

    if (result.response != result.actual) { // only play on incorrect
        const auto soundPath = Utils::getExecutableDirectory() / "assets" / "sounds" / "error.wav";

        PlaySound(soundPath.c_str(), NULL, SND_FILENAME | SND_ASYNC);
    }

    m_results.push_back(result);

    if (m_config.experimentInfo.isEyeTracking) {

        m_eyetracker.clearSamples();
        m_csv.writeRow(
            {
                std::to_string(m_trialIndex),
                std::to_string(result.imageIndex),
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
                m_config.blockInfo.participantID,

                std::to_string(m_config.experimentInfo.fixationTimeMs),                   // fixation time window in MS
                std::to_string(m_config.trials[m_trialIndex].maxMeanPositionPixels),     // max mean position (pixels)
                std::to_string(m_config.trials[m_trialIndex].maxSpreadPixels),           // max spread (pixels) 
                std::to_string(m_config.trials[m_trialIndex].fixationStats.fixation1.meanDistance),
                std::to_string(m_config.trials[m_trialIndex].fixationStats.fixation2.meanDistance),

                std::to_string(m_config.trials[m_trialIndex].fixationStats.fixation1.spread.L),
                std::to_string(m_config.trials[m_trialIndex].fixationStats.fixation1.spread.R),

                std::to_string(m_config.trials[m_trialIndex].fixationStats.fixation2.spread.L),
                std::to_string(m_config.trials[m_trialIndex].fixationStats.fixation2.spread.R)
            }
        );
    }
    else { // when not eye tracking, omit eye tracking related values

        m_csv.writeRow(
            {
                std::to_string(m_trialIndex),
                std::to_string(result.imageIndex),
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
                m_config.blockInfo.participantID,
                // fill remaining gaze / tracking stats (9 columns) that won't be needed for no eye tracking
                "-1", "-1", "-1",
                "-1", "-1",
                "-1", "-1",
                "-1", "-1"
            });
    }
    if (m_decodeThread.joinable()) {
        m_decodeThread.join();
        uploadDecodedTextures(); // after joining, all images are decoded. so we can upload them to the gpu now
    }


    m_trialIndex++;

    if (m_trialIndex >= (int)m_config.trials.size()) {
        m_trialIndex = -1; // this is just for the gaze logging csv file
        m_phase = TrialPhase::Done;
        updateEyeTrackerStage();
        return;
    }

    // load the textures for the upcoming trial (if the response was recorded 
    // before the waitforresponse screen)
    if (m_phase == TrialPhase::ShowSideBySideImages)
    {
        loadTexturesForTrial(m_config.trials[m_trialIndex]);
    }

    showBuffer();

    m_phaseStart = glfwGetTime();
    m_responseStart = m_phaseStart;
    m_flickerShow = false;
    m_flickerLast = m_phaseStart;

}

/// <summary>
/// Skips the current trial, recording it with response = -1.
/// Eye-tracking statistics (the last fixation window of each fixation event) are still saved.
/// The caller is responsible for appending the retrial copy to m_config.trials BEFORE calling this.
/// </summary>
void App::skipTrial()
{
    TrialResult result;

    // No response was provided
    result.reactionTimeMS = -1;
    result.response = -1;

    result.imageName = m_config.trials[m_trialIndex].name;
    result.imageIndex = m_config.trials[m_trialIndex].index;
    result.codec = m_config.trials[m_trialIndex].codec;
    result.positionX_L = m_config.trials[m_trialIndex].fixationCoords.Left.X;
    result.positionY_L = m_config.trials[m_trialIndex].fixationCoords.Left.Y;
    result.positionX_R = m_config.trials[m_trialIndex].fixationCoords.Right.X;
    result.positionY_R = m_config.trials[m_trialIndex].fixationCoords.Right.Y;
    result.foveatLevel = m_config.trials[m_trialIndex].foveatLevel;

    // Translate viewing mode into name data
    switch (m_config.trials[m_trialIndex].viewingMode) {
    case 0: result.viewingMode = "Stereo"; break;
    case 1: result.viewingMode = "Mono Left"; break;
    case 2: result.viewingMode = "Mono Right"; break;
    default: result.viewingMode = "N/A"; break;
    }

    result.actual = m_config.trials[m_trialIndex].flickerIndex;

    m_results.push_back(result);

    // Write CSV row (uses m_trialIndex of the trial being skipped, before the increment below)
    if (m_config.experimentInfo.isEyeTracking) {

        m_eyetracker.clearSamples();

        m_csv.writeRow({
            std::to_string(m_trialIndex),
            std::to_string(result.imageIndex),
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
            std::to_string((int)result.reactionTimeMS),
            m_config.blockInfo.participantID,

            std::to_string(m_config.experimentInfo.fixationTimeMs),
            std::to_string(m_config.trials[m_trialIndex].maxMeanPositionPixels),
            std::to_string(m_config.trials[m_trialIndex].maxSpreadPixels),

            std::to_string(m_config.trials[m_trialIndex].fixationStats.fixation1.meanDistance),
            std::to_string(m_config.trials[m_trialIndex].fixationStats.fixation2.meanDistance),

            std::to_string(m_config.trials[m_trialIndex].fixationStats.fixation1.spread.L),
            std::to_string(m_config.trials[m_trialIndex].fixationStats.fixation1.spread.R),

            std::to_string(m_config.trials[m_trialIndex].fixationStats.fixation2.spread.L),
            std::to_string(m_config.trials[m_trialIndex].fixationStats.fixation2.spread.R)
            });

    }
    else {

        m_csv.writeRow({
            std::to_string(m_trialIndex),
            std::to_string(result.imageIndex),
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
            std::to_string((int)result.reactionTimeMS),
            m_config.blockInfo.participantID,

            "-1", "-1", "-1",
            "-1", "-1",
            "-1", "-1",
            "-1", "-1"
            });
    }

    // make sure no background decode is touching the renderer's staging area
    if (m_decodeThread.joinable())
        m_decodeThread.join();

    // Advance to next trial
    m_trialIndex++;

    if (m_trialIndex >= (int)m_config.trials.size()) {
        m_trialIndex = -1;
        m_phase = TrialPhase::Done;
        updateEyeTrackerStage();
        return;
    }

    // Always start the next trial on its first fixation/interval, even if we
    // were skipped during the second fixation.
    m_interTrialImageIndex = 0;

    // The GPU still holds the skipped trial's images, so load the new trial's textures
    // via the same staged decode -> upload path used in the normal trial flow.
    decodeImagesForTrial(m_config.trials[m_trialIndex]);
    uploadDecodedTextures();

    // showBuffer() after the load: it clears stale gaze samples and resets m_phaseStart
    showBuffer();

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
    const bool leftPressed = state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER] > 0.5f;

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

    if (m_config.experimentInfo.intervalMode == 0) { // 2 interval mode
        m_renderer.decodeAndUploadTexture(TEX_WAIT_L, (assetsDir / "responsescreen0_L.ppm").string());

        m_renderer.decodeAndUploadTexture(TEX_WAIT_R, (assetsDir / "responsescreen0_R.ppm").string());
    }
    else { // single interval mode
        m_renderer.decodeAndUploadTexture(TEX_WAIT_L, (assetsDir / "responsescreen1_L.ppm").string());

        m_renderer.decodeAndUploadTexture(TEX_WAIT_R, (assetsDir / "responsescreen1_R.ppm").string());
    }

    m_renderer.decodeAndUploadTexture(TEX_START_L, (assetsDir / "startscreen_L.ppm").string());

    m_renderer.decodeAndUploadTexture(TEX_START_R, (assetsDir / "startscreen_R.ppm").string());
}


/// <summary>
/// // load the textures for the current trial based on the viewing mode. these are loaded every time the trial is switched.
/// </summary>
/// <param name="img">The paths to the 4 image permuatationss (L, R, Original, Degraded) </param>
void App::loadTexturesForTrial(const TrialInformation& img) {
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
void App::decodeImagesForTrial(const TrialInformation& img) {
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

    if (key == GLFW_KEY_SPACE) {
        app->bypassGaze = true;
    }

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, true);
        return;
    }
    if (key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT)
        app->recordResponse(key);

    if (key == GLFW_KEY_ENTER && app->m_phase == TrialPhase::StartInstructions)
        app->initGame();
}