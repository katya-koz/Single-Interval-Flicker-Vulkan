#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <string>
#include "config.h"
#include "shader.h"
#include "csv.h"

// Each trial steps through these phases in order
enum class TrialPhase {
    StartInstructions, // starting instructions
    ShowImages, // show images side by side (flicker and original)
    WaitForResponse, // black screen — waiting for left/right arrow key
    Done  // all trials are complete
};

struct Texture {
    GLuint id = 0;
    int width;
    int height;
};

struct QuadMesh {
    GLuint vao, vbo;
};



// result recorded for each trial
struct TrialResult {
    std::string imageName;
    int answer;    // first or second (0 , 1)
    int actual; // first or second ( 0, 1)
    double reactionTime;   // seconds from start of WaitForResponse phase
    int index; // order of appearance
    std::string viewingMode;
};

class App {
public:
    App();
    ~App();



    bool init(const std::string& configPath);
    void run();

    // can be edited in config
    //how long each image is shown before advancing ( in seconds)
    double timeoutDuration;

    //how long in between the images (black screen, in seconds)
    double waitTimeoutDuration;

    // image flicker rate in Hz — how many times per second the image swaps (e.g. 10.0 = 10 swaps/sec)
    double flickerRate;

private:
    int m_width;
    int m_height;

    std::string m_title;
    GLFWwindow* m_window = nullptr;
    Config m_config;

    int  m_trialIndex = 0;
    TrialPhase m_phase = TrialPhase::StartInstructions;
    double m_phaseStart = 0.0;
    double m_responseStart = 0.0;
    double m_flickerLast = 0.0;   // glfwGetTime() of last swap
    //bool m_flickerOnOrig = true;  // true = showing orig, false = showing dec
    bool m_flickerShow; // true - dec is on; false - dec is off

    std::vector<TrialResult> m_results;

    //Texture m_texture_L = 0; // current texture on screen (left)
    //Texture m_texture_R = 0;

    // need to load original and decompressed in order to flicker between them (without re-loading the images in every update tick)
    Texture m_texOrig_L;
    Texture m_texDec_L;
    Texture m_texOrig_R;
    Texture m_texDec_R;
    // double up, later for stereo mode.

    // for instrucitons
    Texture m_texStart_L;
    Texture m_texWaitResponse_L;
    Texture m_texStart_R;
    Texture m_texWaitResponse_R;

    // crosshair (fixation point)
    GLuint m_crosshairVAO;
    GLuint m_crosshairVBO;
    Shader m_crosshairShader;


    Shader m_shader;
    QuadMesh m_quadFull; // fulscreen
    QuadMesh m_quadIMG0, m_quadIMG1; // for side by side images
    //Shader m_fovealShader;
    //Shader m_localShader;

    //GLPBO m_pboOrig_L
    //m_pboOrig_R
    //m_pboDec_L
    //m_pboDec_R

    CSV m_csv;
    bool m_prevGamepadA = false;
    bool m_prevGamepadLeft = false;
    bool m_prevGamepadRight = false;
    void update();
    void render();
    void pollGamepad();
    void renderFlickerLayer(Texture origL, Texture origR, Texture decL, Texture decR);

    bool initQuads(int texW, int texH);
    QuadMesh makeQuad(int x, int y, int w, int h);
    bool initCrosshair();
    void advancePhase();
    void loadInstructionsTextures(); // load the response screen and instructions
    void initGame();
    void updateFixationShader();
    void loadTextures(const ImagePaths paths);
    bool loadShaders();
    void initWindow();
    void loadTexture(const std::string& path, Texture& textureID);
    void renderFlickerTexture(Texture texL, Texture texR, int imageIndex); // imageIndex indicates: 0 = left image, 1 = right image
    void renderTexture(Texture texL, Texture texR);
    void renderTexture(Texture texL0, Texture texR0, Texture texL1, Texture texR1);
   
    void renderCrosshair();
    void recordResponse(int key);

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);

    void updateImageShader();
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
};