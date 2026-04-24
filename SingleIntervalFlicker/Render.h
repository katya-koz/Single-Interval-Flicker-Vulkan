#pragma once
#pragma once

#include "rendertypes.h"

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    bool isComplete() const { return graphics.has_value() && present.has_value(); }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};
struct DisplayColorInfo {
    // chromaticities (CIE 1931 xy), matches VkXYColorEXT layout
    float redPrimary[2];
    float greenPrimary[2];
    float bluePrimary[2];
    float whitePoint[2];
    float minLuminance; // nits
    float maxLuminance; // nits (peak, small area)
    float maxFullFrameLuminance; // nits (sustained full-screen)
    bool  isHDR; // true if color space reports PQ/BT.2100
    bool  valid; // false if query failed
};
class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
 
    bool init(GLFWwindow* window, int monitorWidth, int monitorHeight);
 
    void uploadTexture(TextureSlot slot, const std::string& path);

    // render & present 1 frame. blocks on vsync (FIFO)
    void drawFrame(const FrameScene& scene);

    // call before app destroys the window or releases shared resources
    void waitIdle();

private:
    //  per slot texture resourcess
    struct Texture {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    // push constants for the quad pipeline
    struct QuadPushConstants {
        float x0, y0, x1, y1;
        uint32_t texSlot;
    };

    struct CrosshairPushConstants {
        float aspect;
    };


    // vulkan init chain
    void createInstance();
    void setupDebugMessenger();
    void createSurface(GLFWwindow* window);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain(GLFWwindow* window);
    void createImageViews();
    void createRenderPass();
    void createDescriptorSetLayout();
    void allocateDescriptorPool();
    void createGraphicsPipeline();
    void createCrosshairPipeline();
    void createFramebuffers();
    void createCrosshairVertexBuffer();
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();

    // per frame
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex, const FrameScene& scene);
    void drawQuad(VkCommandBuffer cmd, TextureSlot slot, float x0, float y0, float x1, float y1);
    void renderCrosshair(VkCommandBuffer cmd);

    // texture helpers
    void uploadTextureData(Texture& tex, const void* pixels,
        VkFormat fmt, int w, int h);
    void destroyTexture(Texture& tex);
    void updateDescriptorSet(int slot, const Texture& tex);

    // vulkan helpers
    VkShaderModule createShaderModule(const std::vector<uint32_t>& spirv);

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem);
    uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags props);

    void transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout);

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cmd);
    void applyHdrMetadata(GLFWwindow* window);

    // queries
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice dev);

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& a);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& a);
    VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& c);

    // validation
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback( VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT*, void*);

private:
    // size of ONE monitor
    int m_monitorWidth = 0;
    int m_monitorHeight = 0;

    // core vulkan objs
    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue m_presentQueue = VK_NULL_HANDLE;

    // swap chain
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> m_swapImages;
    std::vector<VkImageView> m_swapImageViews;
    VkFormat m_swapFormat = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR m_swapColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D m_swapExtent{};

    // render pass/ frame buffers
    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_framebuffers;

    // pipelines
    VkDescriptorSetLayout m_descSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    VkPipelineLayout m_crosshairLayout = VK_NULL_HANDLE;
    VkPipeline m_crosshairPipeline = VK_NULL_HANDLE;

    //vertex buffer (crosshair only)
    VkBuffer m_crosshairVB = VK_NULL_HANDLE;
    VkDeviceMemory m_crosshairVBMem = VK_NULL_HANDLE;

    // command pool
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_commandBuffers;

    static constexpr int MAX_FRAMES_IN_FLIGHT = 2; //allow CPU to prepare N + 1 frames while GPU is rendering N
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> m_imageAvailable{};
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> m_renderFinished{};
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> m_inFlightFences{};
    uint32_t m_currentFrame = 0;

    // textures and descriptors
    VkDescriptorPool m_descPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_TEXTURES> m_descSets{};
    std::array<Texture, MAX_TEXTURES> m_textures{};
};