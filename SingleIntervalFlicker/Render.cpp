#include "render.h"
#include "shaders.h"

#include <opencv2/opencv.hpp>
#include "Utils.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

// validation layers for debug builds
#ifdef NDEBUG
static constexpr bool ENABLE_VALIDATION = false;
#else
static constexpr bool ENABLE_VALIDATION = true;
#endif

// for querying monitor stats
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#pragma comment(lib, "dxgi.lib")
#endif

static const std::vector<const char*> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"
};

// ensure that the GPU supports swapchains and HDR metadata
static const std::vector<const char*> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_HDR_METADATA_EXTENSION_NAME,
};

// crosshair geometry (t -> thickness, s -> size/ half length)
static constexpr float CROSSHAIR_T = 0.001f;
static constexpr float CROSSHAIR_S = 0.02f;

// these vertices define the shape of the cross hair
static const float CROSSHAIR_VERTS[] = {
    // horizontal bar (left monitor only, indices 0-5)
    -CROSSHAIR_S, -CROSSHAIR_T,
     CROSSHAIR_S, -CROSSHAIR_T,
     CROSSHAIR_S,  CROSSHAIR_T,
    -CROSSHAIR_S, -CROSSHAIR_T,
     CROSSHAIR_S,  CROSSHAIR_T,
    -CROSSHAIR_S,  CROSSHAIR_T,
    // vertical bar (right monitor only, indices 6-11)
    -CROSSHAIR_T, -CROSSHAIR_S,
     CROSSHAIR_T, -CROSSHAIR_S,
     CROSSHAIR_T,  CROSSHAIR_S,
    -CROSSHAIR_T, -CROSSHAIR_S,
     CROSSHAIR_T,  CROSSHAIR_S,
    -CROSSHAIR_T,  CROSSHAIR_S,
};

/// <summary>
/// Renderer lifecycle. Release all memory upon app termination.
/// Disassembles objects in the correct order ( delete children before parents, reverse creation order)
/// </summary>
Renderer::~Renderer() {
    if (m_device) vkDeviceWaitIdle(m_device); // ensure the GPU is not using any resources

    for (auto& tex : m_textures) destroyTexture(tex);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_imageAvailable[i]) vkDestroySemaphore(m_device, m_imageAvailable[i], nullptr);
        if (m_renderFinished[i]) vkDestroySemaphore(m_device, m_renderFinished[i], nullptr);
        if (m_inFlightFences[i]) vkDestroyFence(m_device, m_inFlightFences[i], nullptr);
    }

    if (m_crosshairVB) vkDestroyBuffer(m_device, m_crosshairVB, nullptr);
    if (m_crosshairVBMem) vkFreeMemory(m_device, m_crosshairVBMem, nullptr);

    if (m_descPool) vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
    if (m_descSetLayout) vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

    if (m_pipeline) vkDestroyPipeline(m_device, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    if (m_crosshairPipeline) vkDestroyPipeline(m_device, m_crosshairPipeline, nullptr);
    if (m_crosshairLayout) vkDestroyPipelineLayout(m_device, m_crosshairLayout, nullptr);

    for (auto fb : m_framebuffers) vkDestroyFramebuffer(m_device, fb, nullptr);
    for (auto iv : m_swapImageViews) vkDestroyImageView(m_device, iv, nullptr);

    if (m_renderPass) vkDestroyRenderPass(m_device, m_renderPass, nullptr);
    if (m_swapchain) vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);

    if (m_commandPool) vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    if (m_device) vkDestroyDevice(m_device, nullptr);

    if (ENABLE_VALIDATION && m_debugMessenger) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(m_instance, m_debugMessenger, nullptr);
    }

    if (m_surface)  vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    if (m_instance) vkDestroyInstance(m_instance, nullptr);
}
/// <summary>
/// 
/// </summary>
/// <param name="window"> GLFW window created by App </param>
/// <param name="monitorWidth"> Width of monitors, (they are asssumed to be the same), passed down by App.</param>
/// <param name="monitorHeight"> Height of monitors, (they are asssumed to be the same), passed down by App.</param>
/// <returns> True if successful init, false otherwise. </returns>
bool Renderer::init(GLFWwindow* window, int monitorWidth, int monitorHeight) {
    m_monitorWidth = monitorWidth;
    m_monitorHeight = monitorHeight;

    createInstance(); // root vulkan object
    if (ENABLE_VALIDATION) setupDebugMessenger(); // debug callbacks 
    createSurface(window); // bind vulkan to the GLFW window
    pickPhysicalDevice(); // select GPU
    createLogicalDevice(); // create handles to the GPU 
    createSwapChain(window); // creates frame buffer images to present to screen
    createImageViews(); // views into those swap images
    createRenderPass(); // describes render target layout transitions
    createDescriptorSetLayout(); // how the shaders bind to the textures
    createCommandPool(); // allocates GPU command buffers
    allocateDescriptorPool(); // allocates description sets
    createGraphicsPipeline(); // pipeline for drawing textured quads
    createCrosshairPipeline(); // differnet pipeline for drawing the crosshair/ fixation point
    createFramebuffers(); // wrap image views into frame buffer objects
    createCrosshairVertexBuffer(); // uploadss crosshair geometry to the GPU
    createCommandBuffers(); // creates command buffers,one per frame in flight
    createSyncObjects(); // create semaphores and fences for syncing
    return true;
}

void Renderer::waitIdle() {
    if (m_device) vkDeviceWaitIdle(m_device);
}


/// <summary>
/// Called every frame. Executes the frame loop.
/// </summary>
/// <param name="scene"> Pass through the scene to draw (current images, state, isFlicker, etc) </param>
void Renderer::drawFrame(const FrameScene& scene) {
    // wait until the GPU is ddone with the previous use of this frame slot
    // frames in flight is 2 (by default), so m_currentFrame cycles 0 -> 1 -> 0 -> 1... allows CPU to prepare N+1 while GPU renders N
    //( fences let the CPU know when the GPU has finished its work) CPU -> GPU syncronization
    vkWaitForFences(m_device, 1, &m_inFlightFences[m_currentFrame],VK_TRUE, UINT64_MAX); 
     
    uint32_t imageIndex;
    // asks the swapchain which image should be rendered next. 
    VkResult res = vkAcquireNextImageKHR(
        m_device, 
        m_swapchain, 
        UINT64_MAX, 
        m_imageAvailable[m_currentFrame], 
        VK_NULL_HANDLE, 
        &imageIndex // which image in swapchain to render
    );

    //if the window was resized, return
    if (res == VK_ERROR_OUT_OF_DATE_KHR) return;
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) throw std::runtime_error("vkAcquireNextImageKHR failed");

    // reset the fence
    vkResetFences(m_device, 1, &m_inFlightFences[m_currentFrame]);
    vkResetCommandBuffer(m_commandBuffers[m_currentFrame], 0);

    // re-record the frame's command buffer from scratch based on the current frame scene 
    recordCommandBuffer(m_commandBuffers[m_currentFrame], imageIndex, scene);

    // wait for imageAvailable semaphore, and signal renderFinished when done
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &m_imageAvailable[m_currentFrame];
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_commandBuffers[m_currentFrame];
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &m_renderFinished[m_currentFrame];

    // CPU submits work to the GPU
    if (vkQueueSubmit(m_graphicsQueue, 1, &si,
        m_inFlightFences[m_currentFrame]) != VK_SUCCESS)
        throw std::runtime_error("vkQueueSubmit failed");

    // wait for renderfinished, then tell the swapchain to present the next image
    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &m_renderFinished[m_currentFrame];
    pi.swapchainCount = 1;
    pi.pSwapchains = &m_swapchain;
    pi.pImageIndices = &imageIndex;
    vkQueuePresentKHR(m_presentQueue, &pi);

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT; // advance to next frame slot
}

struct Rect {
    float x0, y0, x1, y1;
};

/// <summary>
/// Map quad vertices from pixel coordinates to NDC
/// </summary>
/// <param name="x"> top left corner of quad, x px</param>
/// <param name="y"> top left corner of quad, y px</param>
/// <param name="w"> width of quad px</param>
/// <param name="h"> height of quad px</param>
/// <param name="screenW"> monitor width px</param>
/// <param name="screenH">monitor height px</param>
/// <returns></returns>
static Rect makeQuadNDC(int x, int y, int w, int h, int screenW, int screenH)
{
    float x0 = (2.0f * x / screenW) - 1.0f;
    float x1 = (2.0f * (x + w) / screenW) - 1.0f;

    float y0 = -(2.0f * y / screenH) + 1.0f;
    float y1 = -(2.0f * (y + h) / screenH) + 1.0f;

    return { x0, y1, x1, y0 };
}
/// <summary>
/// Calcualtes the image rects based on their size. 
/// Images will sit side by side on same monitor, with a gap between them.
/// If image is portrait, then clamp based on image height.
/// If image is landscape, clamp based on image width.
/// </summary>
/// <param name="texW"></param>
/// <param name="texH"></param>
/// <param name="screenW"></param>
/// <param name="screenH"></param>
/// <param name="img0"></param>
/// <param name="img1"></param>
void computeImageRects(
    int texW, int texH,
    int screenW, int screenH,
    Rect& img0, Rect& img1)
{
    float aspectRatio = (float)texW / (float)texH;

    int gap = 60;
    int halfW = screenW / 2;

    int imgW, imgH;

    if (aspectRatio >= 1.0f) {
        imgW = halfW - (int)(gap * 1.5f);
        imgH = (int)(imgW / aspectRatio);
    }
    else {
        imgH = screenH - 2 * gap;
        imgW = (int)(imgH * aspectRatio);
    }

    // clamp
    if (imgW > halfW - (int)(gap * 1.5f)) {
        imgW = halfW - (int)(gap * 1.5f);
        imgH = (int)(imgW / aspectRatio);
    }

    int imgY = (screenH - imgH) / 2;
    int leftX = gap;
    int rightX = halfW + gap / 2;

    // convert to NDC
    img0 = makeQuadNDC(leftX, imgY, imgW, imgH, screenW, screenH);
    img1 = makeQuadNDC(rightX, imgY, imgW, imgH, screenW, screenH);
}

/// <summary>
/// Recieves a frame scene object and translates it to Vulkan commands.
/// </summary>
/// <param name="cmd"> Command buffer </param>
/// <param name="imageIndex"></param>
/// <param name="scene"> Frame Scene object </param>
void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex, const FrameScene& scene)
{
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &bi); // begin the command buffer, start recording

    // render pass (clears frame buffer to black)
    VkClearValue clearColor{};
    clearColor.color = { 0.0f, 0.0f, 0.0f, 1.0f };

    VkRenderPassBeginInfo rpi{};
    rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpi.renderPass = m_renderPass;
    rpi.framebuffer = m_framebuffers[imageIndex];
    rpi.renderArea.offset = { 0, 0 };
    rpi.renderArea.extent = m_swapExtent;
    rpi.clearValueCount = 1;
    rpi.pClearValues = &clearColor;

    vkCmdBeginRenderPass(cmd, &rpi, VK_SUBPASS_CONTENTS_INLINE);

    // sets the viewport
    // set viewport(0) will map NDC to the left half of the window, which is the left edge of the left monitor
    // set viewport(monitorWidth) will map the NDC to the middle of the window, which is the left edge of the right monitor
    auto setViewport = [&](int xOffset) {
        VkViewport vp{};
        vp.x = (float)xOffset;
        vp.y = 0.0f;
        vp.width = (float)m_monitorWidth;
        vp.height = (float)m_monitorHeight;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);

        VkRect2D sc{};
        sc.offset = { xOffset, 0 };
        sc.extent = { (uint32_t)m_monitorWidth, (uint32_t)m_monitorHeight };
        vkCmdSetScissor(cmd, 0, 1, &sc);
    };

    // full screen NDC rect
    const float FX0 = -1.0f, FY0 = -1.0f, FX1 = 1.0f, FY1 = 1.0f;

    // 2 image side by side NDC rects
    Rect left0, left1, right0, right1;
    computeImageRects(m_textures[TEX_ORIG_L].width, m_textures[TEX_ORIG_L].height, m_monitorWidth, m_monitorHeight, left0, left1);

    // since we are shifting the viewport before drawing the right monitor, 
    // the image positions are the same on both monitors.
    right0 = left0;
    right1 = left1;


    // image quads
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    // choose what to display based on the mode passed
    switch (scene.mode) {
        case FrameScene::Mode::ShowImages: {
            // left monitor
            setViewport(0);
            TextureSlot TEX_IMAGE0_L = TEX_ORIG_L;
            TextureSlot TEX_IMAGE1_L = TEX_ORIG_L;


            if (scene.flickerShow) { // flicker either on the first image or the second image
               scene.flickerIndex == 0 ? TEX_IMAGE0_L = TEX_DEC_L : TEX_IMAGE1_L = TEX_DEC_L;
            }
       
            // left monitor, image 0
            drawQuad(cmd, TEX_IMAGE0_L, left0.x0, left0.y0, left0.x1, left0.y1);
            // left monitor image 1
            drawQuad(cmd, TEX_IMAGE1_L, left1.x0, left1.y0, left1.x1, left1.y1);

            // right monitor
            setViewport(m_monitorWidth);

            TextureSlot TEX_IMAGE0_R = TEX_ORIG_R;
            TextureSlot TEX_IMAGE1_R = TEX_ORIG_R;

            if (scene.flickerShow) {
                scene.flickerIndex == 0 ? TEX_IMAGE0_R = TEX_DEC_R : TEX_IMAGE1_R = TEX_DEC_R;
            }

            // right monitor, image 0
            drawQuad(cmd, TEX_IMAGE0_R, right0.x0, right0.y0, right0.x1, right0.y1);
            // right monitor image 1
            drawQuad(cmd, TEX_IMAGE1_R, right1.x0, right1.y0, right1.x1, right1.y1);
            break;
        }
        // draw full screen quads for these modes
        case FrameScene::Mode::StartInstructions:
            setViewport(0);
            drawQuad(cmd, TEX_START_L, FX0, FY0, FX1, FY1);
            setViewport(m_monitorWidth); 
            drawQuad(cmd, TEX_START_R, FX0, FY0, FX1, FY1);
            break;

        case FrameScene::Mode::WaitForResponse:
            setViewport(0);
            drawQuad(cmd, TEX_WAIT_L, FX0, FY0, FX1, FY1);
            setViewport(m_monitorWidth); 
            drawQuad(cmd, TEX_WAIT_R, FX0, FY0, FX1, FY1);
            break;

        case FrameScene::Mode::Blank:
            // clear already handled by render pass
            break;
    }

    // draw the crosshair
    if (scene.drawCrosshair) renderCrosshair(cmd);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}
/// <summary>
/// Draws a single quad (textured rectangle) from 2 corners, (x0,y0) -> (x1,y1)
/// </summary>
/// <param name="cmd"></param>
/// <param name="slot">texture slot</param>
/// <param name="x0">top left corner, x</param>
/// <param name="y0">top left corner, y</param>
/// <param name="x1">bottom right corner, x</param>
/// <param name="y1">bottom right corner,y</param>
void Renderer::drawQuad(VkCommandBuffer cmd, TextureSlot slot, float x0, float y0, float x1, float y1)
{
    // texture selection -> tells fragment shader whcih image to sample from
    vkCmdBindDescriptorSets(
        cmd, 
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipelineLayout, 
        0, 
        1,
        &m_descSets[slot], 
        0, 
        nullptr
    );

    // send draw data to shaders (push constants is a quick ans small way to push ddata to shaders without ussing a buffer)
    QuadPushConstants pc{ x0, y0, x1, y1, (uint32_t)slot};
    vkCmdPushConstants(
        cmd, 
        m_pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(pc), 
        &pc
    );
    
    // 1 quad = 2 triangles = 6 vertices
    // issue the draw call, per vertex 
    vkCmdDraw(cmd, 6, 1, 0, 0);
}


/// <summary>
/// Draws the fixation cross. This is the render pipeline for the crosshair.
/// </summary>
/// <param name="cmd"></param>
void Renderer::renderCrosshair(VkCommandBuffer cmd) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_crosshairPipeline);

    VkBuffer     vbs[] = { m_crosshairVB };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);

    auto setViewport = [&](int xOff) {
        VkViewport vp{ (float)xOff, 0, (float)m_monitorWidth, (float)m_monitorHeight, 0, 1 };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ { xOff, 0 }, { (uint32_t)m_monitorWidth, (uint32_t)m_monitorHeight } };
        vkCmdSetScissor(cmd, 0, 1, &sc);
    };

    CrosshairPushConstants chp{ (float)m_monitorWidth / (float)m_monitorHeight }; // push aspect ratio so that the vertex shader for the crosshair can correct for non square monitors
    vkCmdPushConstants(cmd, m_crosshairLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(chp), &chp);

    // draw differnet parts of the crosshair on different monitors. 
    // that way, when the eyes are correctly converged on the steroscopic mirror setup, the crosshair will be correct
    setViewport(0);
    vkCmdDraw(cmd, 6, 1, 0, 0); // horizontal bar on left monitor

    setViewport(m_monitorWidth);
    vkCmdDraw(cmd, 6, 1, 6, 0); // vertical bar on right monitor
}

/// <summary>
/// Loads images from the disk. Handles PPM images. 
/// </summary>
/// <param name="slot">Texture slot to be loaded in</param>
/// <param name="path">Image path</param>
void Renderer::uploadTexture(TextureSlot slot, const std::string& path) {
    // wait for the gpu to finish before updating any texture slots.
    vkDeviceWaitIdle(m_device);

    // read the PPM max value from the ppm header to determine whether image is HDR
    double ppmMax = 255.0;
    {
        std::ifstream f(path, std::ios::binary);
        if (f.is_open()) {
            std::string magic;
            f >> magic;
            while (f.peek() == '\n') f.get();
            while (f.peek() == '#') f.ignore(4096, '\n');
            std::string w, h, maxval;
            f >> w >> h >> maxval;
            ppmMax = std::stod(maxval);
        }
    }

    // load pixels in with openCV
    cv::Mat src = cv::imread(path, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);
    
    if (src.empty()) throw std::runtime_error("[Renderer] Failed to load image: " + path);

    const bool isHDR = (ppmMax > 255.0); // hdr > 8 bit

    // convert to linear rgb values
    cv::Mat img;
    src.convertTo(img, CV_32F, 1.0 / ppmMax); // normalize to [0,1]
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB); // openCV uses BGR, and vulkan expects RGB
    cv::flip(img, img, -1); // images need to be mirrored (stereoscopic mirror setup), and for some reason they're loaded in upside down, so need to flip on x axis as well to correct for that.
    if (!img.isContinuous()) img = img.clone();

    // choose vulkan format and pack
    VkFormat fmt;
    cv::Mat  upload;

    if (isHDR) {
        cv::Mat rgba;
        cv::cvtColor(img, rgba, cv::COLOR_RGB2RGBA);
        rgba.convertTo(upload, CV_16F);
        fmt = VK_FORMAT_R16G16B16A16_SFLOAT; // HDR format, 16 b half floats for RGB values
    }
    else { // non hdr format
        cv::Mat rgba;
        cv::cvtColor(img, rgba, cv::COLOR_RGB2RGBA);
        rgba.convertTo(upload, CV_8U, 255.0); // 8 bit - rescale back from [0,1] to [0,255]
        fmt = VK_FORMAT_R8G8B8A8_SRGB;
    }

    if (!upload.isContinuous()) upload = upload.clone();

    Texture& tex = m_textures[slot];
    destroyTexture(tex);
    tex.width = upload.cols;
    tex.height = upload.rows;
    uploadTextureData(tex, upload.data, fmt, upload.cols, upload.rows);
    updateDescriptorSet(slot, tex);
}

/// <summary>
/// Staging buffer upload. 
/// Staging buffers are used to efficiently transfer data from CPU -> GPU. A bridge.
/// </summary>
/// <param name="tex"></param>
/// <param name="pixels"></param>
/// <param name="fmt"></param>
/// <param name="w"></param>
/// <param name="h"></param>
void Renderer::uploadTextureData(Texture& tex, const void* pixels,
    VkFormat fmt, int w, int h)
{
    //create staging buffer in host visible memory
    const size_t bpp = (fmt == VK_FORMAT_R16G16B16A16_SFLOAT) ? 8 : 4;
    const VkDeviceSize size = (VkDeviceSize)(w * h) * bpp;

    VkBuffer stageBuf;
    VkDeviceMemory stageMem;
    createBuffer(
        size, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stageBuf, 
        stageMem);

    void* mapped;
    // map gpu memory into cpu address space to write into it directly
    vkMapMemory(m_device, stageMem, 0, size, 0, &mapped);
    //memcpy the pixel data 
    std::memcpy(mapped, pixels, size);
    //then unmap it
    vkUnmapMemory(m_device, stageMem);

    // gpu image
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = { (uint32_t)w, (uint32_t)h, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL; // optimize for gpu, cpu can no longer read it
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; // image wil recieve data from sstaging buffer and be sampled in shaders
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkCreateImage(m_device, &ici, nullptr, &tex.image);

    // get the memory requirements for the image
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(m_device, tex.image, &mr);

    // then allocate the memory (fast GPU memory, not accessible by CPU)
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(m_device, &mai, nullptr, &tex.memory);

    // bind the allocated memory to image
    vkBindImageMemory(m_device, tex.image, tex.memory, 0); 

    // transition image to Transfer_DST_OPTIMAL state to tell vulkan we are about to write into it
    transitionImageLayout(tex.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // copy the staging buffer into gpu image
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { (uint32_t)w, (uint32_t)h, 1 };

    // copies raw pixel data from CPU staging buffer into GPU image memory
    vkCmdCopyBufferToImage(cmd, stageBuf, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    endSingleTimeCommands(cmd);

    // image state, ready for shaders
    transitionImageLayout(tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // image view -> how shaders will interpret this image
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = tex.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    // define how vulkan will interpret the image data (color channels, mip levels etc)
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.layerCount = 1;

    vkCreateImageView(m_device, &vci, nullptr, &tex.view);

    // sampler (how the texture is read in shaders)
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

    // linear filtering smooth the interpolation between texels
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;

    // clamp prevents from wrapping at edges
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    vkCreateSampler(m_device, &sci, nullptr, &tex.sampler);

    // cleanup temp upload resources (reverse creation order)
    vkDestroyBuffer(m_device, stageBuf, nullptr);
    vkFreeMemory(m_device, stageMem, nullptr);
}

/// <summary>
/// Helper to destroy textures
/// </summary>
/// <param name="tex"></param>
void Renderer::destroyTexture(Texture& tex) {
    if (!m_device) return;
    if (tex.sampler) { vkDestroySampler(m_device, tex.sampler, nullptr); tex.sampler = VK_NULL_HANDLE; }
    if (tex.view) { vkDestroyImageView(m_device, tex.view, nullptr);   tex.view = VK_NULL_HANDLE; }
    if (tex.image) { vkDestroyImage(m_device, tex.image, nullptr);      tex.image = VK_NULL_HANDLE; }
    if (tex.memory) { vkFreeMemory(m_device, tex.memory, nullptr);       tex.memory = VK_NULL_HANDLE; }
}

/// <summary>
/// Helper to update the descriptor set
/// </summary>
/// <param name="slot"></param>
/// <param name="tex"></param>
void Renderer::updateDescriptorSet(int slot, const Texture& tex) {
    VkDescriptorImageInfo ii{};
    ii.sampler = tex.sampler;
    ii.imageView = tex.view;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet wr{};
    wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr.dstSet = m_descSets[slot];
    wr.dstBinding = 0;
    wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr.pImageInfo = &ii;
    vkUpdateDescriptorSets(m_device, 1, &wr, 0, nullptr);
}

// Vulkan setup

/// <summary>
/// create the root vulkan object
/// </summary>
void Renderer::createInstance() {
    if (ENABLE_VALIDATION && !checkValidationLayerSupport())
        throw std::runtime_error("Validation layers requested but not available");

    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "FlickerExperiment";
    ai.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    ai.apiVersion = VK_API_VERSION_1_2;

    auto exts = getRequiredExtensions();

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &ai;
    ci.enabledExtensionCount = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCI{};
    if (ENABLE_VALIDATION) {
        ci.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size();
        ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();

        debugCI.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCI.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCI.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        debugCI.pfnUserCallback = debugCallback;
        ci.pNext = &debugCI;
    }

    if (vkCreateInstance(&ci, nullptr, &m_instance) != VK_SUCCESS)
        throw std::runtime_error("vkCreateInstance failed");
}
/// <summary>
/// Debug callbacks
/// </summary>
void Renderer::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    ci.pfnUserCallback = debugCallback;

    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
    if (!fn) throw std::runtime_error("Cannot load vkCreateDebugUtilsMessengerEXT");
    fn(m_instance, &ci, nullptr, &m_debugMessenger);
}

/// <summary>
/// Creates the drawing surface from a GLFW window
/// </summary>
/// <param name="window"></param>
void Renderer::createSurface(GLFWwindow* window) {
    if (glfwCreateWindowSurface(m_instance, window, nullptr, &m_surface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface failed");
}

/// <summary>
/// Pick a Vulkan comaptible GPU to use
/// </summary>
void Renderer::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (!count) throw std::runtime_error("No Vulkan compatible GPU found");

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    for (auto& dev : devices) {
        auto idx = findQueueFamilies(dev);
        auto scs = querySwapChainSupport(dev);
        if (idx.isComplete() && !scs.formats.empty() && !scs.presentModes.empty()) {
            m_physicalDevice = dev;
            break;
        }
    }
    if (!m_physicalDevice) throw std::runtime_error("No suitable GPU found");
}

/// <summary>
/// Create the software to handle to that GPU + queues
/// </summary>
void Renderer::createLogicalDevice() {
    auto idx = findQueueFamilies(m_physicalDevice);
    std::set<uint32_t> uniqueFamilies{ idx.graphics.value(), idx.present.value() };

    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qcis;
    for (auto family : uniqueFamilies) {
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        qcis.push_back(qci);
    }

    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount = (uint32_t)qcis.size();
    ci.pQueueCreateInfos = qcis.data();
    ci.enabledExtensionCount = (uint32_t)DEVICE_EXTENSIONS.size();
    ci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
    ci.pEnabledFeatures = &features;

    if (ENABLE_VALIDATION) {
        ci.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size();
        ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }

    if (vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_device) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDevice failed");

    vkGetDeviceQueue(m_device, idx.graphics.value(), 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, idx.present.value(), 0, &m_presentQueue);
}

// swap chain (HDR10)s

// queries the monitor for stats 
#ifdef _WIN32
static DisplayColorInfo queryPrimaryMonitorHDR(GLFWwindow* window) {
    using Microsoft::WRL::ComPtr;

    DisplayColorInfo info{};
    info.valid = false;

    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        std::cerr << "[Renderer] CreateDXGIFactory1 failed\n";
        return info;
    }

    // prefer the output that contains our window; fall back to adapter 0 / output 0
    HMONITOR targetMonitor = nullptr;
    if (window) {
        HWND hwnd = glfwGetWin32Window(window);
        if (hwnd) targetMonitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY);
    }

    ComPtr<IDXGIOutput6> output6;

    // first pass: find the output matching our window's monitor
    if (targetMonitor) {
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT ai = 0; factory->EnumAdapters1(ai, &adapter) != DXGI_ERROR_NOT_FOUND; ++ai) {
            ComPtr<IDXGIOutput> output;
            for (UINT oi = 0; adapter->EnumOutputs(oi, &output) != DXGI_ERROR_NOT_FOUND; ++oi) {
                DXGI_OUTPUT_DESC desc{};
                if (SUCCEEDED(output->GetDesc(&desc)) && desc.Monitor == targetMonitor) {
                    output.As(&output6);
                    break;
                }
                output.Reset();
            }
            if (output6) break;
            adapter.Reset();
        }
    }

    // fallback: adapter 0, output 0 (typically the primary monitor)
    if (!output6) {
        ComPtr<IDXGIAdapter1> adapter;
        if (SUCCEEDED(factory->EnumAdapters1(0, &adapter))) {
            ComPtr<IDXGIOutput> output;
            if (SUCCEEDED(adapter->EnumOutputs(0, &output))) {
                output.As(&output6);
            }
        }
    }

    if (!output6) {
        std::cerr << "[Renderer] No DXGI output found\n";
        return info;
    }

    DXGI_OUTPUT_DESC1 d{};
    if (FAILED(output6->GetDesc1(&d))) {
        std::cerr << "[Renderer] IDXGIOutput6::GetDesc1 failed\n";
        return info;
    }

    info.redPrimary[0] = d.RedPrimary[0];
    info.redPrimary[1] = d.RedPrimary[1];
    info.greenPrimary[0] = d.GreenPrimary[0];
    info.greenPrimary[1] = d.GreenPrimary[1];
    info.bluePrimary[0] = d.BluePrimary[0];
    info.bluePrimary[1] = d.BluePrimary[1];
    info.whitePoint[0] = d.WhitePoint[0];
    info.whitePoint[1] = d.WhitePoint[1];
    info.minLuminance = d.MinLuminance;
    info.maxLuminance = d.MaxLuminance;
    info.maxFullFrameLuminance = d.MaxFullFrameLuminance;
    info.isHDR = (d.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020);
    info.valid = true;
    return info;
}
#endif // _WIN32


/// <summary>
/// Chooses the swap format in order of priority
/// HDR10 first, fall back on sRGB
/// </summary>
/// <param name="available"></param>
/// <returns></returns>
VkSurfaceFormatKHR Renderer::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available)
{
    for (auto& f : available) {
        if (f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
            f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT); return f;
    }
    for (auto& f : available) {
        if (f.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 &&
            f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) return f;
    }
    for (auto& f : available) {
        if (f.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
            f.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) return f;
    }
    return available[0];
}

/// <summary>
/// Enables v sync
/// </summary>
/// <param name="available"></param>
/// <returns></returns>
VkPresentModeKHR Renderer::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& available)
{
    //for (auto m : available)
    //    if (m == VK_PRESENT_MODE_FIFO_KHR) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

/// <summary>
/// Choose the swap chain resolution
/// </summary>
/// <param name="window"></param>
/// <param name="caps"></param>
/// <returns></returns>
VkExtent2D Renderer::chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& caps)
{
    if (caps.currentExtent.width != UINT32_MAX) return caps.currentExtent;

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    return {
        std::clamp((uint32_t)w, caps.minImageExtent.width,  caps.maxImageExtent.width),
        std::clamp((uint32_t)h, caps.minImageExtent.height, caps.maxImageExtent.height)
    };
}

/// <summary>
/// Queries the OS for the primary monitors stats, to record in metadata.
/// </summary>
/// <param name="window"></param>
void Renderer::applyHdrMetadata(GLFWwindow* window) {
    // query the primary monitor: on non windows this returns an invalid info. 
    // this experiemnet is expected to run with 2 identical monitors.
    #ifdef _WIN32
        DisplayColorInfo disp = queryPrimaryMonitorHDR(window);
    #else
        DisplayColorInfo disp{};
    #endif

    // hdr must be active for this experiment. fail otherwise
    if (!disp.valid || !disp.isHDR) {
        Utils::FatalError("[Renderer] HDR10 not active on primary and secondary monitor. Enable 'Use HDR' in Windows Display Settings");
    }

    // some drivers report 0 for luminance fields even when primaries are valid
    // treat that as a separate fallback case
    const bool lumValid = disp.maxLuminance > 0.0f;

    VkHdrMetadataEXT meta{};
    meta.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT;

    // primaries + white point — always queried at this point (we threw above if invalid)
    meta.displayPrimaryRed = { disp.redPrimary[0],   disp.redPrimary[1] };
    meta.displayPrimaryGreen = { disp.greenPrimary[0], disp.greenPrimary[1] };
    meta.displayPrimaryBlue = { disp.bluePrimary[0],  disp.bluePrimary[1] };
    meta.whitePoint = { disp.whitePoint[0],   disp.whitePoint[1] };

    // luminance, queried if reported, default otherwise
    if (lumValid) {
        meta.maxLuminance = disp.maxLuminance;
        meta.minLuminance = disp.minLuminance;

        // dont claim brighter content than the panel can show
        const float fullFrame = (disp.maxFullFrameLuminance > 0.0f) ? disp.maxFullFrameLuminance : disp.maxLuminance * 0.4f;
        meta.maxContentLightLevel = disp.maxLuminance;
        meta.maxFrameAverageLightLevel = fullFrame;
    }
    else { // fallback
        meta.maxLuminance = 1000.0f;
        meta.minLuminance = 0.001f;
        meta.maxContentLightLevel = 1000.0f;
        meta.maxFrameAverageLightLevel = 400.0f;
    }

    //debug message
    std::cout << "[Renderer] HDR metadata: queried primaries, "
        << (lumValid ? "queried" : "fallback") << " luminance — "
        << "peak=" << meta.maxLuminance << " nits, "
        << "fullframe=" << meta.maxFrameAverageLightLevel << " nits\n";

    // apply to the swapchain
    auto fn = (PFN_vkSetHdrMetadataEXT)
        vkGetDeviceProcAddr(m_device, "vkSetHdrMetadataEXT");
    if (fn) {
        fn(m_device, 1, &m_swapchain, &meta);
    }
    else {
        std::cerr << "[Renderer] vkSetHdrMetadataEXT not available; "
            "HDR metadata not applied\n";
    }
}

/// <summary>
/// Creates the swap chain (with helpers)
/// </summary>
/// <param name="window"></param>
void Renderer::createSwapChain(GLFWwindow* window) {
    auto scs = querySwapChainSupport(m_physicalDevice);
    auto sfmt = chooseSwapSurfaceFormat(scs.formats);
    auto smode = chooseSwapPresentMode(scs.presentModes);
    auto extent = chooseSwapExtent(window, scs.capabilities);

    m_swapFormat = sfmt.format;
    m_swapColorSpace = sfmt.colorSpace;
    m_swapExtent = extent;

    uint32_t imgCount = scs.capabilities.minImageCount + 1;
    if (scs.capabilities.maxImageCount > 0)
        imgCount = (std::min)(imgCount, scs.capabilities.maxImageCount);

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = m_surface;
    ci.minImageCount = imgCount;
    ci.imageFormat = sfmt.format;
    ci.imageColorSpace = sfmt.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    auto idx = findQueueFamilies(m_physicalDevice);
    uint32_t families[] = { idx.graphics.value(), idx.present.value() };
    if (idx.graphics != idx.present) {
        ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices = families;
    }
    else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    ci.preTransform = scs.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = smode;
    ci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(m_device, &ci, nullptr, &m_swapchain) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSwapchainKHR failed");

    applyHdrMetadata(window);

    // retrieve swap chain images
    uint32_t n;
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &n, nullptr);
    m_swapImages.resize(n);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &n, m_swapImages.data());
}

void Renderer::createImageViews() {
    m_swapImageViews.resize(m_swapImages.size());
    for (size_t i = 0; i < m_swapImages.size(); ++i) {
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = m_swapImages[i];
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = m_swapFormat;
        ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ci.subresourceRange.levelCount = 1;
        ci.subresourceRange.layerCount = 1;
        vkCreateImageView(m_device, &ci, nullptr, &m_swapImageViews[i]);
    }
}
/// <summary>
/// Render target layout transitions
/// </summary>
void Renderer::createRenderPass() {
    VkAttachmentDescription att{};
    att.format = m_swapFormat;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpi{};
    rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpi.attachmentCount = 1;
    rpi.pAttachments = &att;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &sub;
    rpi.dependencyCount = 1;
    rpi.pDependencies = &dep;

    if (vkCreateRenderPass(m_device, &rpi, nullptr, &m_renderPass) != VK_SUCCESS)
        throw std::runtime_error("vkCreateRenderPass failed");
}

void Renderer::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 1;
    li.pBindings = &b;

    if (vkCreateDescriptorSetLayout(m_device, &li, nullptr, &m_descSetLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorSetLayout failed");
}

void Renderer::allocateDescriptorPool() {
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps.descriptorCount = MAX_TEXTURES;

    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &ps;
    pi.maxSets = MAX_TEXTURES;

    if (vkCreateDescriptorPool(m_device, &pi, nullptr, &m_descPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorPool failed");

    std::vector<VkDescriptorSetLayout> layouts(MAX_TEXTURES, m_descSetLayout);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = m_descPool;
    ai.descriptorSetCount = MAX_TEXTURES;
    ai.pSetLayouts = layouts.data();

    if (vkAllocateDescriptorSets(m_device, &ai, m_descSets.data()) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateDescriptorSets failed");
}

VkShaderModule Renderer::createShaderModule(const std::vector<uint32_t>& spirv) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spirv.size() * sizeof(uint32_t);
    ci.pCode = spirv.data();

    VkShaderModule mod;
    if (vkCreateShaderModule(m_device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("vkCreateShaderModule failed");
    return mod;
}

void Renderer::createGraphicsPipeline() {
    auto vertSpv = compileGLSL(QUAD_VERT_GLSL, shaderc_vertex_shader, "quad.vert");
    auto fragSpv = compileGLSL(QUAD_FRAG_GLSL, shaderc_fragment_shader, "quad.frag");

    VkShaderModule vertMod = createShaderModule(vertSpv);
    VkShaderModule fragMod = createShaderModule(fragSpv);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vis{};
    vis.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ias{};
    ias.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vps{};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode = VK_CULL_MODE_NONE;
    rast.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAtt.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAtt;

    std::vector<VkDynamicState> dynStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = (uint32_t)dynStates.size();
    dyn.pDynamicStates = dynStates.data();

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(QuadPushConstants);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &m_descSetLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcRange;

    vkCreatePipelineLayout(m_device, &pli, nullptr, &m_pipelineLayout);

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vis;
    pci.pInputAssemblyState = &ias;
    pci.pViewportState = &vps;
    pci.pRasterizationState = &rast;
    pci.pMultisampleState = &ms;
    pci.pColorBlendState = &blend;
    pci.pDynamicState = &dyn;
    pci.layout = m_pipelineLayout;
    pci.renderPass = m_renderPass;

    if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pci,
        nullptr, &m_pipeline) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines failed");

    vkDestroyShaderModule(m_device, vertMod, nullptr);
    vkDestroyShaderModule(m_device, fragMod, nullptr);
}

void Renderer::createCrosshairPipeline() {
    auto vertSpv = compileGLSL(CROSSHAIR_VERT_GLSL, shaderc_vertex_shader, "ch.vert");
    auto fragSpv = compileGLSL(CROSSHAIR_FRAG_GLSL, shaderc_fragment_shader, "ch.frag");

    VkShaderModule vertMod = createShaderModule(vertSpv);
    VkShaderModule fragMod = createShaderModule(fragSpv);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,   vertMod, "main" };
    stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "main" };

    VkVertexInputBindingDescription bind{};
    bind.binding = 0;
    bind.stride = 2 * sizeof(float);
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attr{};
    attr.location = 0;
    attr.binding = 0;
    attr.format = VK_FORMAT_R32G32_SFLOAT;
    attr.offset = 0;

    VkPipelineVertexInputStateCreateInfo vis{};
    vis.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vis.vertexBindingDescriptionCount = 1;
    vis.pVertexBindingDescriptions = &bind;
    vis.vertexAttributeDescriptionCount = 1;
    vis.pVertexAttributeDescriptions = &attr;

    VkPipelineInputAssemblyStateCreateInfo ias{};
    ias.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vps{};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode = VK_CULL_MODE_NONE;
    rast.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAtt.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAtt;

    std::vector<VkDynamicState> dynStates{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = (uint32_t)dynStates.size();
    dyn.pDynamicStates = dynStates.data();

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcr.size = sizeof(CrosshairPushConstants);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(m_device, &pli, nullptr, &m_crosshairLayout);

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vis;
    pci.pInputAssemblyState = &ias;
    pci.pViewportState = &vps;
    pci.pRasterizationState = &rast;
    pci.pMultisampleState = &ms;
    pci.pColorBlendState = &blend;
    pci.pDynamicState = &dyn;
    pci.layout = m_crosshairLayout;
    pci.renderPass = m_renderPass;

    if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pci,
        nullptr, &m_crosshairPipeline) != VK_SUCCESS)
        throw std::runtime_error("createCrosshairPipeline failed");

    vkDestroyShaderModule(m_device, vertMod, nullptr);
    vkDestroyShaderModule(m_device, fragMod, nullptr);
}

void Renderer::createFramebuffers() {
    m_framebuffers.resize(m_swapImageViews.size());
    for (size_t i = 0; i < m_swapImageViews.size(); ++i) {
        VkFramebufferCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass = m_renderPass;
        fi.attachmentCount = 1;
        fi.pAttachments = &m_swapImageViews[i];
        fi.width = m_swapExtent.width;
        fi.height = m_swapExtent.height;
        fi.layers = 1;
        vkCreateFramebuffer(m_device, &fi, nullptr, &m_framebuffers[i]);
    }
}

void Renderer::createCrosshairVertexBuffer() {
    const VkDeviceSize size = sizeof(CROSSHAIR_VERTS);

    VkBuffer       stageBuf;
    VkDeviceMemory stageMem;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stageBuf, stageMem);

    void* mapped;
    vkMapMemory(m_device, stageMem, 0, size, 0, &mapped);
    std::memcpy(mapped, CROSSHAIR_VERTS, size);
    vkUnmapMemory(m_device, stageMem);

    createBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_crosshairVB, m_crosshairVBMem);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy    cp{ 0, 0, size };
    vkCmdCopyBuffer(cmd, stageBuf, m_crosshairVB, 1, &cp);
    endSingleTimeCommands(cmd);

    vkDestroyBuffer(m_device, stageBuf, nullptr);
    vkFreeMemory(m_device, stageMem, nullptr);
}

void Renderer::createCommandPool() {
    auto idx = findQueueFamilies(m_physicalDevice);
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = idx.graphics.value();
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(m_device, &ci, nullptr, &m_commandPool);
}

void Renderer::createCommandBuffers() {
    m_commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = m_commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    vkAllocateCommandBuffers(m_device, &ai, m_commandBuffers.data());
}

VkCommandBuffer Renderer::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = m_commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(m_device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void Renderer::endSingleTimeCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    vkQueueSubmit(m_graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_graphicsQueue);
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);
}

void Renderer::createSyncObjects() {
    VkSemaphoreCreateInfo si{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo     fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkCreateSemaphore(m_device, &si, nullptr, &m_imageAvailable[i]);
        vkCreateSemaphore(m_device, &si, nullptr, &m_renderFinished[i]);
        vkCreateFence(m_device, &fi, nullptr, &m_inFlightFences[i]);
    }
}


//************* Buffer and Memory helpers 
void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem)
{
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(m_device, &bi, nullptr, &buf);

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(m_device, buf, &mr);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, props);
    vkAllocateMemory(m_device, &ai, nullptr, &mem);
    vkBindBufferMemory(m_device, buf, mem, 0);
}

uint32_t Renderer::findMemoryType(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("findMemoryType: no suitable type found");
}

void Renderer::transitionImageLayout(VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout)
{
    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkPipelineStageFlags srcStage, dstStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
        newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else {
        throw std::runtime_error("Unsupported layout transition");
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    endSingleTimeCommands(cmd);
}

// ****************** queries

QueueFamilyIndices Renderer::findQueueFamilies(VkPhysicalDevice dev) {
    QueueFamilyIndices idx;
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            idx.graphics = i;

        VkBool32 present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, m_surface, &present);
        if (present) idx.present = i;

        if (idx.isComplete()) break;
    }
    return idx;
}

SwapChainSupportDetails Renderer::querySwapChainSupport(VkPhysicalDevice dev) {
    SwapChainSupportDetails d;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, m_surface, &d.capabilities);

    uint32_t n;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, m_surface, &n, nullptr);
    d.formats.resize(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, m_surface, &n, d.formats.data());

    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, m_surface, &n, nullptr);
    d.presentModes.resize(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, m_surface, &n, d.presentModes.data());

    return d;
}

// *********************** validation helpers

bool Renderer::checkValidationLayerSupport() {
    uint32_t count;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> available(count);
    vkEnumerateInstanceLayerProperties(&count, available.data());

    for (const char* name : VALIDATION_LAYERS) {
        bool found = false;
        for (auto& layer : available)
            if (std::strcmp(name, layer.layerName) == 0) { found = true; break; }
        if (!found) return false;
    }
    return true;
}

std::vector<const char*> Renderer::getRequiredExtensions() {
    uint32_t glfwCount;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);
    std::vector<const char*> exts(glfwExts, glfwExts + glfwCount);
    exts.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    if (ENABLE_VALIDATION)
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    return exts;
}

VKAPI_ATTR VkBool32 VKAPI_CALL Renderer::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*)
{
    std::cerr << "[VK] " << data->pMessage << "\n";
    return VK_FALSE;
}