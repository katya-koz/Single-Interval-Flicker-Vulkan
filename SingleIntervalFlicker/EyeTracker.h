#pragma once
#include <GLFW/glfw3.h>
#include <vector>
#include <mutex>

#include "tobii_research_eyetracker.h"
#include "tobii_research_streams.h"
#include "tobii_research.h"

#define DEBUG_MOUSE_GAZE // use mouse point rather than eye tracker

class EyeTracker
{
public:
    void start(TobiiResearchEyeTracker* eyetracker);
    void stop();

    void startCollection();
    void stopCollection();

    std::vector<TobiiResearchGazeData> getSamples();

    void clearSamples();
    bool enterCalibration(TobiiResearchEyeTracker* eyetracker);
    bool collectCalibrationPoint(TobiiResearchEyeTracker* eyetracker, float x, float y);
    bool finishCalibration(TobiiResearchEyeTracker* eyetracker);

    #ifdef DEBUG_MOUSE_GAZE
        void updateMouseGaze(GLFWwindow* window, int monitorWidth);
    #endif

private:
    static void gaze_data_callback(
        TobiiResearchGazeData* gaze_data,
        void* user_data
    );
   

    std::mutex m_mutex;
    std::vector<TobiiResearchGazeData> m_samples;
    bool m_collecting = false; // flag to save to samples, or not.
};