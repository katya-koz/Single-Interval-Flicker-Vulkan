#pragma once

#include "render.h" 
#include "rendertypes.h"
#include "config.h"
#include "csv.h"
#include "utils.h"
#include <thread>
#define GLFW_INCLUDE_NONE
#define DEBUG_MOUSE_GAZE
#include <GLFW/glfw3.h>

#include <string>
#include <vector>
#include "EyeTracker.h"


enum class TrialPhase {
    StartInstructions,
    ShowSideBySideImages,
    ShowFullFieldImage,
    ShowBuffer,
    WaitForResponse,
    Done
};


struct TrialResult {
    std::string codec;
    std::string foveatLevel;
    std::string imageName;
    int actual;
    int positionX_L;
    int positionY_L;
    int positionX_R;
    int positionY_R;
    std::string viewingMode;
    int response;
    int reactionTimeMS;
    
};

class App {
public:
    App() = default;
    ~App();

    App(const App&) = delete;
    App& operator=(const App&) = delete;

    bool init(const std::string& configPath, std::string& inputPath);
    void run();
     
private:
    void initGame();
    void update();
    void advancePhase();
    void recordResponse(int key);
    void pollGamepad();
    void showBuffer();
    void showNextImageInTrial();
    std::thread m_decodeThread;


    // translate current phase + flicker state into a scene description
    // so that the renderer can draw
    FrameScene buildScene() const;

    // wrap the renderer to load textures
    void loadInstructionsTextures();
    void loadTexturesForTrial(const ImagePaths& img);

    void decodeImagesForTrial(const ImagePaths& img);
    void decodeImageForUpload(TextureSlot slot, const std::string& path);
    void uploadDecodedTexture(TextureSlot slot);
    void uploadDecodedTextures();
    // glfw callbacks
    static void keyCallback(GLFWwindow*, int, int, int, int);
    static void framebufferSizeCallback(GLFWwindow*, int, int) {} // unused

private:
    EyeTracker m_eyetracker;
    
    GLFWwindow* m_window = nullptr;
    int m_monitorWidth = 0;
    int m_monitorHeight = 0;

    Renderer m_renderer; // backend renderer
    Config m_config;
    int m_trialIndex = 0;

    // used when in two interval mode. tracks whether 
    // the first or second image within a trial has been shown. 0 or 1
    int m_interTrialImageIndex = 0; 

    std::vector<TrialResult> m_results;

    // experiment timing
    double timeoutDuration = 0.0;
    double flickerRate = 0.0;
    double waitTimeoutDuration = 0.0;

    // for the state machine
    TrialPhase m_phase = TrialPhase::StartInstructions;
    double m_phaseStart = 0.0;
    double m_responseStart = 0.0;

    // flicker bool (is this frame a flicker frame?)
    double m_flickerLast = 0.0;
    bool   m_flickerShow = false;

    double m_flickerInterval = 0; // how long to show image before flickering 

    // gamepad edge detection
    bool m_prevGamepadA = false;
    bool m_prevGamepadLeft = false;
    bool m_prevGamepadRight = false;

    CSV m_csv;
};