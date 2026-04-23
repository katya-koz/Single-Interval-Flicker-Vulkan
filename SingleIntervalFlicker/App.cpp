#include "app.h"
#include <iostream>
#include "Utils.h"
#include "csv.h"
#include <mmsystem.h>
#include <Windows.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#include <GLFW/glfw3native.h>
#pragma comment(lib, "opengl32.lib")
#define WGL_COLORSPACE_EXT 0x309D
#define WGL_COLORSPACE_SCRGB_LINEAR_EXT 0x3101

typedef BOOL(WINAPI* PFNWGLSWAPINTERVALEXTPROC)(int interval);
typedef BOOL(WINAPI* PFNWGLCHOOSEPIXELFORMATARBPROC)(HDC, const int*, const FLOAT*, UINT, int*, UINT*);
typedef HGLRC(WINAPI* PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC, HGLRC, const int*);
typedef BOOL(WINAPI* PFNWGLGETPIXELFORMATATTRIBIVARBPROC)(HDC, int, int, UINT, const int*, int*);

// WGL colorspace extension
typedef BOOL(WINAPI* PFNWGLSETPIXELFORMATCOLORSPACEEXTPROC)(HDC hdc, int colorspace);
#pragma comment(lib, "winmm.lib")

// SHADERS -> to be moved into their own files.

static const std::string VERT_SRC = R"(
#version 460 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
uniform bool uMirror;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = uMirror ? vec2(1.0 - aTexCoord.x, aTexCoord.y) : aTexCoord; 
}
)";



static const std::string FRAG_SRC = R"(
#version 330 core
out vec4 FragColor;
  
in vec2 TexCoord;

uniform sampler2D hdrBuffer;

void main()
{             
    FragColor = vec4(texture(hdrBuffer, TexCoord).rgb, 1.0);
}  
)";


static const std::string CROSSHAIR_VERT_SRC = R"(
#version 460 core
layout (location = 0) in vec2 aPos;

uniform float uAspect;

void main() {
    gl_Position = vec4(aPos.x / uAspect, aPos.y, 0.0, 1.0);
}
)";

static const std::string CROSSHAIR_FRAG_SRC = R"(
#version 460 core
out vec4 FragColor;

uniform vec4 uColor;

void main() {
    FragColor = uColor;
}
)";


float t = 0.001f; // thickness (represents half the thickness)
float s = 0.02f; // size (distance from center to edge of crosshair)

// crosshair to be used as fixation point. horizontal line on left screen, right has veritcal lines. correctly converging eyes will align the cross hairs
float crosshair[] = {
    // horizontal rectangle
    -s, -t,
     s, -t,
     s,  t,

    -s, -t,
     s,  t,
    -s,  t,

    // vertical rectangle
    -t, -s,
     t, -s,
     t,  s,

    -t, -s,
     t,  s,
    -t,  s,
};


static const float QUAD_VERTS[] = {
    // pos          // uv
    -1.0f,  1.0f,   0.0f, 1.0f,   // top-left
    -1.0f, -1.0f,   0.0f, 0.0f,   // bottom-left
     1.0f, -1.0f,   1.0f, 0.0f,   // bottom-right

    -1.0f,  1.0f,   0.0f, 1.0f,   // top-left
     1.0f, -1.0f,   1.0f, 0.0f,   // bottom-right
     1.0f,  1.0f,   1.0f, 1.0f,   // top-right
};

App::App() {  }

App::~App() {

    GLuint textures[] = { m_texOrig_L.id, m_texDec_L.id, m_texStart_L.id, m_texWaitResponse_L.id, m_texOrig_R.id, m_texDec_R.id, m_texStart_R.id, m_texWaitResponse_R.id }; // textures are batch deleted
    glDeleteTextures(8, textures);
    
    if (m_quadFull.vao) glDeleteVertexArrays(1, &m_quadFull.vao);
    if (m_quadFull.vbo) glDeleteBuffers(1, &m_quadFull.vbo);
    if (m_crosshairVAO) glDeleteVertexArrays(1, &m_crosshairVAO);
    if (m_crosshairVBO) glDeleteBuffers(1, &m_crosshairVBO);

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

bool App::initCrosshair() {
    glGenVertexArrays(1, &m_crosshairVAO);
    glGenBuffers(1, &m_crosshairVBO);

    glBindVertexArray(m_crosshairVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_crosshairVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(crosshair), crosshair, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    return m_crosshairShader.load(CROSSHAIR_VERT_SRC, CROSSHAIR_FRAG_SRC);
}

bool App::loadShaders() {
    if (!m_shader.load(VERT_SRC, FRAG_SRC)) return false; // all variants use this shader (full image)

    return true;
}

void App::initWindow() {
    GLFWmonitor* monitor = glfwGetPrimaryMonitor(); // assume the same resolution across the monitors
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    m_width = mode->width;
    m_height = mode->height;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // request float frame buffers for hdr imaging
    glfwWindowHint(GLFW_RED_BITS, 16);
    glfwWindowHint(GLFW_GREEN_BITS, 16);
    glfwWindowHint(GLFW_BLUE_BITS, 16);
    glfwWindowHint(GLFW_ALPHA_BITS, 16);

    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

    m_window = glfwCreateWindow(m_width * 2, m_height, "Flicker Experiment", nullptr, nullptr);

    glfwMakeContextCurrent(m_window);
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    glfwSetKeyCallback(m_window, keyCallback);


    // Call this after glfwMakeContextCurrent in initWindow()
    HDC hdc = GetDC(glfwGetWin32Window(m_window));

    auto wglSetPixelFormatColorspaceEXT =
        (PFNWGLSETPIXELFORMATCOLORSPACEEXTPROC)wglGetProcAddress("wglSetPixelFormatColorspaceEXT");

    if (wglSetPixelFormatColorspaceEXT) {
        wglSetPixelFormatColorspaceEXT(hdc, WGL_COLORSPACE_SCRGB_LINEAR_EXT);
    }
    else {
        OutputDebugStringW(L"WGL HDR colorspace extension not available\n");
    }

}

void App::initGame() {
    m_trialIndex = 0;
    m_phase = TrialPhase::ShowImages;
    m_phaseStart = glfwGetTime();
}

bool App::init(const std::string& configPath) {
    // config init
    if (!m_config.load(configPath))   return false;
    if (m_config.trials.empty()) {
        Utils::FatalError("[App] No trials in config.");
        return false;
    }

    timeoutDuration = m_config.imageTime;
    flickerRate = m_config.flickerRate;
    waitTimeoutDuration = m_config.waitTime;

    // randomize order of trials + flickers
    Utils::ShuffleTrials(m_config.trials);
    Utils::ShuffleFlickers(m_config.trials);

    // init glfw
    if (!glfwInit()) {
        Utils::FatalError("[App] Failed to initialize GLFW");
        return false;
    }

    initWindow();

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        Utils::FatalError("[App] Failed to initialize GLAD");
        return false;
    }

    glViewport(0, 0, m_width, m_height);
    glfwSwapInterval(1);

   // std::string FOVEAL_FRAG_SRC = Utils::ReadFile("shaders/foveal_fragment.glsl"); // fix util to orngaize shaders better :3
    //build shaders
    if (!loadShaders()) return false;

    //quad geometry building
    m_quadFull = makeQuad(0, 0, m_width, m_height); // fullscreen quad
    if (!m_quadFull.vao) return false;
    if (!initCrosshair()) return false;

    //allocate texture slots
    glGenTextures(1, &m_texOrig_L.id);
    glGenTextures(1, &m_texDec_L.id);
    glGenTextures(1, &m_texStart_L.id);
    glGenTextures(1, &m_texWaitResponse_L.id);
    glGenTextures(1, &m_texOrig_R.id);
    glGenTextures(1, &m_texDec_R.id);
    glGenTextures(1, &m_texStart_R.id);
    glGenTextures(1, &m_texWaitResponse_R.id);

    loadInstructionsTextures();
    loadTextures(m_config.trials[0]); // loads the first 2 images in the trial

    std::string variantName;
    updateImageShader(); // used for non flicker condition, will load no matter the variant
    updateFixationShader();
  
    // load the csv
    m_csv.init(m_config.participantID, m_config.participantAge, m_config.participantGender, "Flicker Paradigm", variantName, { "Index", "Image", "Viewing Mode", "Answer", "Actual", "Reaction Time (s)" }, m_config.outputDirectory.string());

    //init phase
    m_phase = TrialPhase::StartInstructions;
    m_phaseStart = glfwGetTime();
    return true;
}

bool App::initQuads(int texW, int texH) {
    float aspectRatio = (float)texW / (float)texH;
    int gap = 60;
    int halfW = m_width / 2;

    int imgW, imgH;

    if (aspectRatio >= 1.0f) {
        imgW = halfW - (int)(gap * 1.5f);
        imgH = (int)(imgW / aspectRatio);
    }
    else {
        imgH = m_height - 2 * gap;
        imgW = (int)(imgH * aspectRatio);
    }

    // clamp so image never overflows its half
    if (imgW > halfW - (int)(gap * 1.5f)) {
        imgW = halfW - (int)(gap * 1.5f);
        imgH = (int)(imgW / aspectRatio);
    }

    int imgY = (m_height - imgH) / 2;  // vertically centered
    int leftX = gap;
    int rightX = halfW + gap / 2;

    // left monitor
    m_quadIMG0 = makeQuad(leftX, imgY, imgW, imgH);
    m_quadIMG1 = makeQuad(rightX, imgY, imgW, imgH);

    //// right monitor (offset by m_width)
    //m_quadR0 = makeQuad(leftX + m_width, imgY, imgW, imgH);
    //m_quadR1 = makeQuad(rightX + m_width, imgY, imgW, imgH);

    return true;
}

// quad init
QuadMesh App::makeQuad(int x, int y, int w, int h) {
    // Convert pixel rect to NDC
    float x0 = (2.0f * x / m_width) - 1.0f;
    float x1 = (2.0f * (x + w) / m_width) - 1.0f;
    float y0 = (2.0f * y / m_height) - 1.0f;
    float y1 = (2.0f * (y + h) / m_height) - 1.0f;

    float verts[] = {
        // x     y     u     v
        x0,  y1,  0.0f, 1.0f,  // top-left
        x0,  y0,  0.0f, 0.0f,  // bottom-left
        x1,  y0,  1.0f, 0.0f,  // bottom-right

        x0,  y1,  0.0f, 1.0f,  // top-left
        x1,  y0,  1.0f, 0.0f,  // bottom-right
        x1,  y1,  1.0f, 1.0f,  // top-right
    };

    QuadMesh q;
    glGenVertexArrays(1, &q.vao);
    glGenBuffers(1, &q.vbo);

    glBindVertexArray(q.vao);
    glBindBuffer(GL_ARRAY_BUFFER, q.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    // position: location 0
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // texcoord: location 1
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    return q;
}

// main loop
void App::run() {
    using clock = std::chrono::high_resolution_clock;

    const double targetFrameTime = 1.0 / m_config.targetFPS;
    auto nextFrameTime = clock::now();

    while (!glfwWindowShouldClose(m_window) && m_phase != TrialPhase::Done) {
        update();
        render();
        glfwSwapBuffers(m_window);
        glfwPollEvents();

        nextFrameTime += std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(targetFrameTime)
        );
        std::this_thread::sleep_until(nextFrameTime);
    }
}

// update loop

void App::update() {
    const double now = glfwGetTime();
    const double elapsed = now - m_phaseStart;
    const ImagePaths& img = m_config.trials[m_trialIndex];

    if (m_phase == TrialPhase::ShowImages) {
        if (elapsed >= timeoutDuration) {
            advancePhase();
            return;
        }
        const double flickerInterval = 1.0 / flickerRate; // seconds per swap
        if (now - m_flickerLast >= flickerInterval) {
            m_flickerLast = now;
            m_flickerShow = !m_flickerShow;
        }

    }

    pollGamepad();
}


// rendering


void App::render() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (m_phase == TrialPhase::ShowImages)
    {
        renderTexture(m_texOrig_L, m_texOrig_L, m_texOrig_R, m_texOrig_R); // renders the texture on both images (side by side)
        if (m_flickerShow) {
            m_config.trials[m_trialIndex].flickerIndex == 0  ? renderTexture(m_texDec_L, m_texOrig_L, m_texDec_R, m_texOrig_R) : renderTexture(m_texOrig_L, m_texDec_L, m_texOrig_R, m_texDec_R);
        }
        
        //renderFlickerTexture(m_texDec_L, m_texDec_R, m_config.trials[m_trialIndex].flickerIndex); // renders the flicker texture over one image
    }

    else if (m_phase == TrialPhase::StartInstructions)
    {
        renderTexture(m_texStart_L, m_texStart_R);
    }

    else if (m_phase == TrialPhase::WaitForResponse)
    {
        renderTexture(m_texWaitResponse_L, m_texWaitResponse_R);
    }

    // render the fixation point (cross hair)
    renderCrosshair();
}


void App::renderTexture(Texture texL, Texture texR) {
    m_shader.use();

    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(m_quadFull.vao);

    // left monitor
    glViewport(0, 0, m_width, m_height);
    glBindTexture(GL_TEXTURE_2D, texL.id);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // right monitor (assuming same width as left)
    glViewport(m_width, 0, m_width, m_height);
    glBindTexture(GL_TEXTURE_2D, texR.id);
    
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// renders the degraded photo over top of the image corresponding to the imageIndex, 0 = left image, 1 = right image
void App::renderFlickerTexture(Texture texL, Texture texR, int imageIndex) {
    m_shader.use();

    glActiveTexture(GL_TEXTURE0);

    auto draw = [](QuadMesh& q, GLuint texId) {
        glBindVertexArray(q.vao);
        glBindTexture(GL_TEXTURE_2D, texId);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        };

    if (imageIndex == 0) {
        // left monitor
        glViewport(0, 0, m_width, m_height);
        draw(m_quadIMG0, texL.id);
        //right
        glViewport(m_width, 0, m_width, m_height);
        draw(m_quadIMG0, texR.id);

    }
    else {
        // left monitor
        glViewport(0, 0, m_width, m_height);
        draw(m_quadIMG1, texL.id);
        //right
        glViewport(m_width, 0, m_width, m_height);
        draw(m_quadIMG1, texR.id);
    }
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}
int m_lastTexW = -1, m_lastTexH = -1;

// renders 2 images side by side per monitor L/R refer to monitor, 0 and 1 refer to image order 
void App::renderTexture(Texture texL0, Texture texL1,  Texture texR0, Texture texR1) {
    // rebuild quads only when image size changes
    if (texL0.width != m_lastTexW || texL0.height != m_lastTexH) {
        initQuads(texL0.width, texL0.height);
        m_lastTexW = texL0.width;
        m_lastTexH = texL0.height;
    }

    m_shader.use();
    glActiveTexture(GL_TEXTURE0);

    auto draw = [](QuadMesh& q, GLuint texId) {
        glBindVertexArray(q.vao);
        glBindTexture(GL_TEXTURE_2D, texId);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    };


    glViewport(0, 0, m_width, m_height); // set viewport to left monitor
    draw(m_quadIMG0, texL0.id);
    draw(m_quadIMG1, texL1.id);

    glViewport(m_width, 0, m_width, m_height); // set viewport to right monitor
    draw(m_quadIMG0, texR0.id);
    draw(m_quadIMG1, texR1.id);

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}


void App::renderCrosshair() {
    m_crosshairShader.use();

    glBindVertexArray(m_crosshairVAO);
    // render horizontal cross hair on left
    glViewport(0, 0, m_width, m_height);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    // render vertical cross hair on right
    glViewport(m_width, 0, m_width, m_height);
    glDrawArrays(GL_TRIANGLES, 6, 12);

    glBindVertexArray(0);
}

// advance to next image set by preloading 
void App::advancePhase() {
    // only called when timer expires during ShowImages
    m_phase = TrialPhase::WaitForResponse;
    m_phaseStart = glfwGetTime();
    m_responseStart = m_phaseStart;

    // preload next trial's images
    if ((m_trialIndex + 1) < (int)m_config.trials.size()) {
        loadTextures(m_config.trials[m_trialIndex + 1]);
    }
}

void App::recordResponse(int key) {
    // only accept input during ShowImages or WaitForResponse
    if (m_phase != TrialPhase::ShowImages && m_phase != TrialPhase::WaitForResponse) return;

    TrialResult result;
    result.imageName = m_config.trials[m_trialIndex].name;
    result.answer = key == GLFW_KEY_LEFT ? 0 : 1;
    result.actual = m_config.trials[m_trialIndex].flickerIndex;
    result.index = m_trialIndex;

    result.answer == result.actual
        ? PlaySound(TEXT("sounds/Success.wav"), NULL, SND_FILENAME | SND_ASYNC)
        : PlaySound(TEXT("sounds/error.wav"), NULL, SND_FILENAME | SND_ASYNC);

    switch (m_config.trials[m_trialIndex].viewingMode) {
    case 0: result.viewingMode = "Stereo"; break;
    case 1: result.viewingMode = "Left";   break;
    case 2: result.viewingMode = "Right";  break;
    default: result.viewingMode = "N/A";   break;
    }

    result.reactionTime = glfwGetTime() - m_responseStart;
    m_results.push_back(result);

    std::vector<std::string> resultRow = {
        std::to_string(result.index),
        result.imageName,
        result.viewingMode,
        std::to_string(result.answer),
        std::to_string(result.actual),
        std::to_string(result.reactionTime)
    };
    m_csv.writeRow(resultRow);

    m_trialIndex++;

    if (m_trialIndex >= (int)m_config.trials.size()) {
        m_phase = TrialPhase::Done;
        return;
    }

    // if they answered early (during ShowImages), preload next trial now
    if (m_phase == TrialPhase::ShowImages) {
        loadTextures(m_config.trials[m_trialIndex]);
    }
    // if WaitForResponse already preloaded in advancePhase, skip reload

    m_phase = TrialPhase::ShowImages;
    m_phaseStart = glfwGetTime();
    m_responseStart = m_phaseStart;
    m_flickerShow = false;
    m_flickerLast = m_phaseStart;
}

void App::loadInstructionsTextures() {
   
    loadTexture("responsescreen_L.ppm", m_texWaitResponse_L);
    loadTexture("startscreen_L.ppm", m_texStart_L);

    loadTexture("responsescreen_R.ppm", m_texWaitResponse_R);
    loadTexture("startscreen_R.ppm", m_texStart_R);

}

// loads the original and dec textures, based on stereo settings in config
// 0 = stereo
// 1 = left only
// 2 = right only

void App::loadTextures(const ImagePaths img) {

    switch (img.viewingMode) {

    case 0: // stereo
        loadTexture(img.L_orig.string(), m_texOrig_L);
        loadTexture(img.R_orig.string(), m_texOrig_R);

        loadTexture(img.L_dec.string(), m_texDec_L);
        loadTexture(img.R_dec.string(), m_texDec_R);
        break;
    case 1: // left only
        loadTexture(img.L_orig.string(), m_texOrig_L);
        loadTexture(img.L_orig.string(), m_texOrig_R);

        loadTexture(img.L_dec.string(), m_texDec_L);
        loadTexture(img.L_dec.string(), m_texDec_R);

        break;
    case 2: // right only
        loadTexture(img.R_orig.string(), m_texOrig_L);
        loadTexture(img.R_orig.string(), m_texOrig_R);

        loadTexture(img.R_dec.string(), m_texDec_L);
        loadTexture(img.R_dec.string(), m_texDec_R);
        break;

    default:
        Utils::FatalError("[App] viewing mode is not valid. Must be 0, 1, or 2. Is : " + img.viewingMode);
    }
}


void App::loadTexture(const std::string& path, Texture& texture)
{
    // parse the max value from header
    double ppmMax = 255.0;
    {
        std::ifstream f(path, std::ios::binary);
        if (f.is_open()) {
            std::string magic;
            f >> magic;
            // skip comments
            while (f.peek() == '\n') f.get();
            while (f.peek() == '#') f.ignore(4096, '\n');
            std::string w, h, maxval;
            f >> w >> h >> maxval;
            ppmMax = std::stod(maxval);
        }
    }

    cv::Mat src = cv::imread(path, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);

    if (src.empty())
        throw std::runtime_error("Failed to load image: " + path);

    cv::Mat img;
    bool isHDR = (ppmMax > 255.0); 


    if (src.depth() == CV_8U)
    {
        src.convertTo(img, CV_32F, 1.0 / ppmMax);
    }
    else if (src.depth() == CV_16U)
    {
        src.convertTo(img, CV_32F, 1.0 / ppmMax);

    }
    else
    {
        src.convertTo(img, CV_32F);
    }



    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::flip(img, img, 0);
    if (!img.isContinuous()) img = img.clone();

    if (isHDR)
        //img *= (203.0f / 80.0f); // scale to scRGB
        //img *= 1.5f;

    glBindTexture(GL_TEXTURE_2D, texture.id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (!isHDR)
    {
        std::wstring msg = L"This image is not HDR:  " + std::wstring(path.begin(), path.end()) + L"\n";
        OutputDebugStringW(msg.c_str());
        cv::Mat upload;
        img.convertTo(upload, CV_8UC3, 255.0f);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, upload.cols, upload.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, upload.data);
    }
    else
    {
        std::wstring msg = L"This image IS HDR:  " + std::wstring(path.begin(), path.end()) + L"\n";
        OutputDebugStringW(msg.c_str());
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, img.cols, img.rows, 0, GL_RGB, GL_FLOAT, img.data);
    }

    texture.width = img.cols;
    texture.height = img.rows;

    glBindTexture(GL_TEXTURE_2D, 0);
}



/// <summary>
/// Update variables within the regular image shader texture
/// </summary>
void App::updateImageShader() {
    m_shader.use();
    m_shader.setInt("uTexture", 0);
    m_shader.setBool("uMirror", true); // mirror 
    m_shader.setFloat("uPaperWhiteNits", 203.0f);
}

/// <summary>
/// Update the fixation shader
/// </summary>
void App::updateFixationShader() {
    m_crosshairShader.use();
    m_crosshairShader.setVec4("uColor", 1.0f, 1.0f, 1.0f, 1.0f);
    m_crosshairShader.setFloat("uAspect", (float)m_width / (float)m_height);
}

// GLFW callbacks
void App::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// key presses
// add gamepad support
void App::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, true);
        return;
    }

    if (key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT)
        app->recordResponse(key);

    if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
    {
        if (app->m_phase == TrialPhase::StartInstructions)
        {
            app->initGame();
            GLuint textures[] = {app->m_texStart_L.id,  app->m_texStart_R.id }; // delete start textures. not needed anymore.
            glDeleteTextures(2, textures);
        }
    }

}
void App::pollGamepad() {
    GLFWgamepadstate state;
    if (!glfwGetGamepadState(GLFW_JOYSTICK_1, &state))
        return;

    // A button → Enter
    bool aPressed = state.buttons[GLFW_GAMEPAD_BUTTON_A];
    if (aPressed && !m_prevGamepadA) {
        if (m_phase == TrialPhase::StartInstructions) {
            initGame();
            GLuint textures[] = { m_texStart_L.id, m_texStart_R.id };
            glDeleteTextures(2, textures);
        }
    }
    m_prevGamepadA = aPressed;

    // x button for left arrow
    bool leftPressed = state.buttons[GLFW_GAMEPAD_BUTTON_X];
    if (leftPressed && !m_prevGamepadLeft)
        recordResponse(GLFW_KEY_LEFT);
    m_prevGamepadLeft = leftPressed;

    // b button for right arrow
    bool rightPressed = state.buttons[GLFW_GAMEPAD_BUTTON_B];
    if (rightPressed && !m_prevGamepadRight)
        recordResponse(GLFW_KEY_RIGHT);
    m_prevGamepadRight = rightPressed;
}