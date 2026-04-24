#pragma once
#pragma once

#include <cstdint>
#include <string>

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

// What App asks the renderer to put on screen this frame.
// The renderer translates this into Vulkan draw calls.
struct FrameScene {
    enum class Mode {
        StartInstructions,   // TEX_START_L / TEX_START_R
        ShowImages,          // TEX_ORIG_L/R, plus optional degraded overlay
        WaitForResponse,     // TEX_WAIT_L / TEX_WAIT_R
        Blank                // clear to black, crosshair only
    };

    Mode mode = Mode::Blank;

    // Only used when mode == ShowImages:
    //  flickerShow == true --> overlay flicker image on top of originals
    //  flickerIndex == 0 --> degraded on image0 slot, original on image1 slot
    //  flickerIndex != 0 --> degraded on image1 slot, original on image0 slot
    bool flickerShow = false;
    int  flickerIndex = 0;

    // crosshair is always drawn
    bool drawCrosshair = true;
};