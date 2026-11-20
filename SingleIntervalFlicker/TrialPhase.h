#pragma once
enum class TrialPhase {
    StartInstructions,         // 0     TEX_START_L / TEX_START_R
    ShowBuffer,                // 1     Show the buffer grey screen between images within same trial
    ShowImage,                 // 2     fullscreen TEX_ORIG_L/R — this interval holds the original
    ShowFlickerImage,          // 3     fullscreen TEX_DEC_L/R — this interval holds the degraded stim. overlay original on flicker times
    WaitForResponse,           // 4     TEX_WAIT_L / TEX_WAIT_R
    Done,                      // 5
    ShowSideBySideImages,  // TEX_ORIG_L/R, plus optional degraded overlay
    Blank                      // clear to black, fixationPoint only
};

inline const char* stageToString(TrialPhase s)
{
    switch (s)
    {
        case TrialPhase::StartInstructions:
            return "Start Instructions";
        case TrialPhase::ShowBuffer:
            return "Fixation";
        case TrialPhase::ShowImage:
            return "Stimulus";
        case TrialPhase::ShowFlickerImage:
            return "Flicker Stimulus";
        case TrialPhase::WaitForResponse:
            return "Waiting for Response";
        case TrialPhase::Done:
            return "Done";
        case TrialPhase::Blank:
            return "Blank";
        case TrialPhase::ShowSideBySideImages:
            return "Single Interval Image";
            return "Unknown";
    }
}