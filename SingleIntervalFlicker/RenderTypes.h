#pragma once
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

//********
// These are shared types (shared between app.cpp and render.cpp)
//********

// The app uses these symbolic names to tell the renderer which image to upload and draw.
// The renderer owns the actual VkImage/VkImageView/VkSampler behind each slot
enum TextureSlot : int {
    TEX_ORIG_L = 0,
    TEX_ORIG_R,
    TEX_DEC_L,
    TEX_DEC_R,
    TEX_START_L,
    TEX_START_R,
    TEX_WAIT_L,
    TEX_WAIT_R,
    MAX_TEXTURES
};


//after opencv loads image (cpu intensive task, to be done on another thread)
// 
struct DecodedImage {
    std::vector<uint8_t> pixels;
    VkFormat format = VK_FORMAT_UNDEFINED;
    int width = 0;
    int height = 0;
};


struct Coords {
    int X = 0;
    int Y = 0;
};

struct FixationCoordinates {
    Coords Left = Coords();
    Coords Right = Coords();
};

// Normalized [0,1] screen-space point (0,0 = top-left, 1,1 = bottom-right).
// Used for the calibration target so this shared header doesn't need to
// depend on TobiiResearchNormalizedPoint2D / the Tobii SDK headers.
struct NormalizedPoint2D {
    float x = 0.5f;
    float y = 0.5f;
};

// What App asks the renderer to put on screen this frame.
// The renderer translates this into Vulkan draw calls.
struct FrameScene {
    enum class Mode {
        StartInstructions,         // TEX_START_L / TEX_START_R
        ShowSingleIntervalImages,  // TEX_ORIG_L/R, plus optional degraded overlay
        ShowFlickerImage,          // fullscreen TEX_DEC_L/R — this interval holds the degraded stim. overlay original on flicker times
        ShowImage,                 // fullscreen TEX_ORIG_L/R — this interval holds the original
        WaitForResponse,           // TEX_WAIT_L / TEX_WAIT_R
        ShowBuffer,                // Show the buffer grey screen between images within same trial
        Blank                      // clear to black, fixationPoint only
        ,Calibration
    };

    Mode mode = Mode::Blank;

    // Only used when mode == ShowSingleIntervalImages:
    //  flickerShow == true --> overlay flicker image on top of originals
    //  flickerIndex == 0 --> degraded on image0 slot, original on image1 slot
    //  flickerIndex != 0 --> degraded on image1 slot, original on image0 slot
    bool flickerShow = false;
    int  flickerIndex = 0;

    // fixation is always drawn
    bool drawFixationPoint = true;

    FixationCoordinates fixationPointCoords = FixationCoordinates();

    // Only used when mode == Calibration:
    NormalizedPoint2D calibrationTarget = NormalizedPoint2D();  // where to draw the target dot, normalized [0,1]
    float calibrationProgress = 0.0f;                           // 0..1, e.g. drives a shrinking/filling animation before a sample is taken
};