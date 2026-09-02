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

struct Coords {
    int X = 0;
    int Y = 0;
};

struct FixationCoordinates {
    Coords Left = Coords();
    Coords Right = Coords();
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
    
};

