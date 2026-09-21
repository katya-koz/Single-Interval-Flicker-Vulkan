#include "EyeTracker.h"
#include <cstdio>
#include "tobii_research_eyetracker.h"
#include "tobii_research_streams.h"
#include "tobii_research.h"

void EyeTracker::gaze_data_callback(
    TobiiResearchGazeData* gaze_data,
    void* user_data)
{
    auto* eyeTracker = static_cast<EyeTracker*>(user_data);

    std::lock_guard<std::mutex> lock(eyeTracker->m_mutex);

    if (!eyeTracker->m_collecting)
        return;

    eyeTracker->m_samples.push_back(*gaze_data);
}

void EyeTracker::start(TobiiResearchEyeTracker* eyetracker)
{
    char* serial_number = nullptr;

    tobii_research_get_serial_number(eyetracker,&serial_number);

    printf("Subscribing to gaze data for eye tracker with serial number %s.\n", serial_number);

    tobii_research_free_string(serial_number);

    TobiiResearchStatus status = tobii_research_subscribe_to_gaze_data(eyetracker,EyeTracker::gaze_data_callback,this);

    if (status != TOBII_RESEARCH_STATUS_OK)
        return;
}

void EyeTracker::startCollection()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    m_samples.clear();
    m_collecting = true;
}

void EyeTracker::stopCollection()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    m_collecting = false;
}

std::vector<TobiiResearchGazeData> EyeTracker::getSamples()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    return m_samples;
}

void EyeTracker::clearSamples()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    m_samples.clear();
}

#ifdef DEBUG_MOUSE_GAZE

    #include <GLFW/glfw3.h>

    void EyeTracker::updateMouseGaze(GLFWwindow* window, int monitorWidth)
    {
        double mouseX;
        double mouseY;

        glfwGetCursorPos(window, &mouseX, &mouseY);

        int height;
        glfwGetWindowSize(window, nullptr, &height);

        TobiiResearchGazeData gazeData{};

        double x = mouseX / static_cast<double>(monitorWidth); // single monitor width
        double y = mouseY / static_cast<double>(height);

        gazeData.left_eye.gaze_point.position_on_display_area.x = x;
        gazeData.left_eye.gaze_point.position_on_display_area.y = y;

        gazeData.right_eye.gaze_point.position_on_display_area.x = x;
        gazeData.right_eye.gaze_point.position_on_display_area.y = y;

        std::lock_guard<std::mutex> lock(m_mutex);

        if (!m_collecting)
            return;

        m_samples.push_back(gazeData);
    }

#endif