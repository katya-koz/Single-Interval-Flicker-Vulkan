#include "EyeTracker.h"
#include <cstdio>
#include <iomanip>
#include <sstream>
#include "tobii_research_eyetracker.h"
#include "tobii_research_streams.h"
#include "tobii_research.h"
#include "tobii_research_calibration.h"
#include "Utils.h"
void EyeTracker::init(float gazePrecision, float viewingDistance, const std::string& participantId,
    int participantAge, char participantGender, int blockNumber, int sessionNumber,
    int groupNumber, const std::string& outputDirectory) {
    gazePrecisionCm = gazePrecision;
    viewingDistanceCm = viewingDistance;

    if (!participantId.empty())
        openCsvLog(participantId, participantAge, participantGender, blockNumber, sessionNumber,
            groupNumber, outputDirectory);
}

EyeTracker::~EyeTracker() { closeCsvLog(); }

void EyeTracker::openCsvLog(const std::string& participantId, int participantAge,
    char participantGender, int blockNumber, int sessionNumber, int groupNumber,
    const std::string& outputDirectory) {

    static const std::vector<std::string> gazeHeaders = { "system_time_stamp", "device_time_stamp",
        "left_gaze_x", "left_gaze_y", "left_gaze_valid", "left_pupil_diameter", "left_pupil_valid",
        "right_gaze_x", "right_gaze_y", "right_gaze_valid", "right_pupil_diameter",
        "right_pupil_valid" };

    m_csv = std::make_unique<CSV>();
    m_csv->init(participantId, participantAge, participantGender, blockNumber, sessionNumber,
        groupNumber, gazeHeaders, outputDirectory, "gaze");

    m_writerRunning.store(true, std::memory_order_release);
    m_writerThread = std::thread(&EyeTracker::writerThreadMain, this);
}

void EyeTracker::closeCsvLog() {
    if (!m_writerRunning.exchange(false, std::memory_order_acq_rel)) return; // wasn't running

    m_queueCv.notify_all();
    if (m_writerThread.joinable()) m_writerThread.join();

    if (m_csv) m_csv->close();
    m_csv.reset();
}

void EyeTracker::writerThreadMain() {
    std::vector<TobiiResearchGazeData> batch;

    for (;;) {
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCv.wait(lock, [this] {
                return !m_writeQueue.empty() || !m_writerRunning.load(std::memory_order_acquire);
                });
            if (m_writeQueue.empty() && !m_writerRunning.load(std::memory_order_acquire)) break;
            batch.swap(m_writeQueue);
        }
        for (const auto& g : batch) m_csv->writeRow(gazeDataToRow(g));
        batch.clear();
    }
}

std::vector<std::string> EyeTracker::gazeDataToRow(const TobiiResearchGazeData& g) {
    auto num = [](double v) {
        std::ostringstream ss;
        ss << std::setprecision(9) << v;
        return ss.str();
        };
    return { std::to_string(g.system_time_stamp), std::to_string(g.device_time_stamp),
        num(g.left_eye.gaze_point.position_on_display_area.x),
        num(g.left_eye.gaze_point.position_on_display_area.y),
        std::to_string(static_cast<int>(g.left_eye.gaze_point.validity)),
        num(g.left_eye.pupil_data.diameter),
        std::to_string(static_cast<int>(g.left_eye.pupil_data.validity)),
        num(g.right_eye.gaze_point.position_on_display_area.x),
        num(g.right_eye.gaze_point.position_on_display_area.y),
        std::to_string(static_cast<int>(g.right_eye.gaze_point.validity)),
        num(g.right_eye.pupil_data.diameter),
        std::to_string(static_cast<int>(g.right_eye.pupil_data.validity)) };
}

void EyeTracker::gaze_data_callback(TobiiResearchGazeData* gaze_data, void* user_data) {
    auto* eyeTracker = static_cast<EyeTracker*>(user_data);

    {
        std::lock_guard<std::mutex> lock(eyeTracker->m_mutex);
        if (eyeTracker->m_collecting) eyeTracker->m_samples.push_back(*gaze_data);
    }

    // Always log to CSV, independent of m_collecting. We only hand the raw
    // sample off to the writer thread here -- this callback runs on the
    // Tobii SDK's own thread and must stay cheap, no file IO.
    if (eyeTracker->m_writerRunning.load(std::memory_order_acquire)) {
        {
            std::lock_guard<std::mutex> lock(eyeTracker->m_queueMutex);
            eyeTracker->m_writeQueue.push_back(*gaze_data);
        }
        eyeTracker->m_queueCv.notify_one();
    }
}

void EyeTracker::start(TobiiResearchEyeTracker* eyetracker) {
    char* serial_number = nullptr;

    tobii_research_get_serial_number(eyetracker, &serial_number);

    printf("Subscribing to gaze data for eye tracker with serial number %s.\n", serial_number);

    tobii_research_free_string(serial_number);

    TobiiResearchStatus status = tobii_research_subscribe_to_gaze_data(eyetracker, EyeTracker::gaze_data_callback, this);

    if (status != TOBII_RESEARCH_STATUS_OK) return;
}

void EyeTracker::stop(TobiiResearchEyeTracker* eyetracker) {
    tobii_research_unsubscribe_from_gaze_data(eyetracker, EyeTracker::gaze_data_callback);
}

void EyeTracker::startCollection() {
    std::lock_guard<std::mutex> lock(m_mutex);

    m_samples.clear();
    m_collecting = true;
}

void EyeTracker::stopCollection() {
    std::lock_guard<std::mutex> lock(m_mutex);

    m_collecting = false;
}

std::vector<TobiiResearchGazeData> EyeTracker::getSamples() {
    std::lock_guard<std::mutex> lock(m_mutex);

    return m_samples;
}

void EyeTracker::clearSamples() {
    std::lock_guard<std::mutex> lock(m_mutex);

    m_samples.clear();
}

bool EyeTracker::enterCalibration(TobiiResearchEyeTracker* eyetracker) {
    return tobii_research_screen_based_calibration_enter_calibration_mode(eyetracker) ==
        TOBII_RESEARCH_STATUS_OK;
}

bool EyeTracker::collectCalibrationPoint(TobiiResearchEyeTracker* eyetracker, float x, float y) {
    TobiiResearchStatus status =
        tobii_research_screen_based_calibration_collect_data(eyetracker, x, y);
    if (status != TOBII_RESEARCH_STATUS_OK)
        status = tobii_research_screen_based_calibration_collect_data(
            eyetracker, x, y); // one retry, per Tobii's own example

    return status == TOBII_RESEARCH_STATUS_OK;
}

bool EyeTracker::finishCalibration(TobiiResearchEyeTracker* eyetracker) {
    TobiiResearchCalibrationResult* result = nullptr;
    TobiiResearchStatus status =
        tobii_research_screen_based_calibration_compute_and_apply(eyetracker, &result);

    bool success = (status == TOBII_RESEARCH_STATUS_OK && result &&
        result->status == TOBII_RESEARCH_CALIBRATION_SUCCESS);

    if (result) {
        // result->calibration_points[i].calibration_samples[s].left_eye/right_eye.validity
        // tells you whether that sample was usable per-eye, if you want to flag/redo weak points
        // instead of trusting compute_and_apply's overall status blindly.
        tobii_research_free_screen_based_calibration_result(result);
    }

    tobii_research_screen_based_calibration_leave_calibration_mode(eyetracker);
    return success;
}

bool EyeTracker::gazeWithinError(
    int pixelError,
    int leftFixationX,
    int leftFixationY,
    int rightFixationX,
    int rightFixationY
) {

    const TobiiResearchGazeData& sample = m_samples.back();

    return Utils::isGazeValid(
        sample.left_eye.gaze_point.position_on_display_area.x,
        sample.left_eye.gaze_point.position_on_display_area.y,
        sample.right_eye.gaze_point.position_on_display_area.x,
        sample.right_eye.gaze_point.position_on_display_area.y,
        pixelError,
        leftFixationX,
        leftFixationY,
        rightFixationX,
        rightFixationY
    );
}

    #ifdef DEBUG_MOUSE_GAZE

        #include <GLFW/glfw3.h>

        void EyeTracker::updateMouseGaze(GLFWwindow* window, int monitorWidth) {
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

            if (!m_collecting) return;

            m_samples.push_back(gazeData);
        }

    #endif