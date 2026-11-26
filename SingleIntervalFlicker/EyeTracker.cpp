#include "EyeTracker.h"
#include <cstdio>
#include <iomanip>
#include <sstream>
#include "tobii_research_eyetracker.h"
#include "tobii_research_streams.h"
#include "tobii_research.h"
#include "tobii_research_calibration.h"
#include "Utils.h"
#include <algorithm>
#include <cmath>


void EyeTracker::init( ExperimentInformation experimentInfo, DisplayInfo displayInfo, BlockInformation blockInfo ) {
    m_experimentInfo = experimentInfo;
    m_displayInfo = displayInfo;

    if (!blockInfo.participantID.empty())
    {
        openCsvLog( blockInfo);
    }
}

EyeTracker::~EyeTracker() {
    closeCsvLog(); 
}

void EyeTracker::gaze_data_callback(TobiiResearchGazeData* gaze_data, void* user_data) {
    auto* et = static_cast<EyeTracker*>(user_data);

    {
        std::lock_guard<std::mutex> lock(et->m_mutex);
        et->m_samples.push(*gaze_data); // rolling fixation window
        if (et->m_collecting) et->m_trialSamples.push_back(*gaze_data);
    }

    if (et->m_writerRunning.load(std::memory_order_acquire)) {
        {
            std::lock_guard<std::mutex> lock(et->m_queueMutex);
            et->m_writeQueue.push_back(*gaze_data);
        }
        et->m_queueCv.notify_one();
    }
}

void EyeTracker::startCollection() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_samples.clear(); // so stale samples from the last trial can't pass the fixation check
    m_collecting = true;
}

bool EyeTracker::getLatestSample(TobiiResearchGazeData& out) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_samples.latest(out);
}

void EyeTracker::clearSamples() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_samples.clear();
    m_trialSamples.clear();
}

void EyeTracker::markExperimentStart() {
    int64_t now = 0;
    if (tobii_research_get_system_time_stamp(&now) == TOBII_RESEARCH_STATUS_OK)
        m_originUs.store(now, std::memory_order_release);
}

void EyeTracker::openCsvLog(BlockInformation blockInfo) {

    static const std::vector<std::string> gazeHeaders = { 
        "system_time_stamp",
        "device_time_stamp",
        "left_gaze_x",
        "left_gaze_y",
        "left_gaze_valid",
        "left_pupil_diameter",
        "left_pupil_valid", 
        "right_gaze_x", 
        "right_gaze_y", 
        "right_gaze_valid", 
        "right_pupil_diameter",
        "right_pupil_valid",
        "current_stage",
        "current_stage_name",
        "trial index",
        "experiment_time_s",
        "event_number" // either 0 or 1 (stim 0/1 or fixation 0/1. -1 if not applicable to current phase)
    };

    m_csv = std::make_unique<CSV>();
    m_csv->init(blockInfo, gazeHeaders, false, "gaze"); // has metadata header = false, tag = "gaze"

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

std::vector<std::string> EyeTracker::gazeDataToRow(const TobiiResearchGazeData& g) const {
    auto num = [](double v) {
        std::ostringstream ss;
        ss << std::setprecision(9) << v;
        return ss.str();
        };

    // seconds since markExperimentStart(); blank until it has been called 
    std::string expTime;
    const int64_t origin = m_originUs.load(std::memory_order_acquire);
    if (origin != kOriginUnset) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(6)
            << (static_cast<double>(g.system_time_stamp - origin) / 1e6);
        expTime = ss.str();
    }


    return { 
        std::to_string(g.system_time_stamp), std::to_string(g.device_time_stamp),
        num(g.left_eye.gaze_point.position_on_display_area.x),
        num(g.left_eye.gaze_point.position_on_display_area.y),
        std::to_string(static_cast<int>(g.left_eye.gaze_point.validity)),
        num(g.left_eye.pupil_data.diameter),
        std::to_string(static_cast<int>(g.left_eye.pupil_data.validity)),
        num(g.right_eye.gaze_point.position_on_display_area.x),
        num(g.right_eye.gaze_point.position_on_display_area.y),
        std::to_string(static_cast<int>(g.right_eye.gaze_point.validity)),
        num(g.right_eye.pupil_data.diameter),
        std::to_string(static_cast<int>(g.right_eye.pupil_data.validity)),
        std::to_string(static_cast<int>(currentStage)),   // integer value
        std::string(stageToString(currentStage)),         // name
        std::to_string(currentTrialIndex),
        expTime,
        std::to_string(currentEventNumber),
    };
}

void EyeTracker::updateStage(TrialPhase stage, int trialIndex) {
    currentStage = stage;
    currentTrialIndex = trialIndex;
}

/// <summary>
/// Event number signifies if this is the 1st or 2nd stim/fixation within the trial.
/// 0 - first
/// 1 - second
/// -1 - not applicable to current phase
/// </summary>
/// <param name="eventNumber"></param>
/// 
void EyeTracker::updateEventNumber(int eventNumber) {
    currentEventNumber = eventNumber;
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


void EyeTracker::stopCollection() {
    std::lock_guard<std::mutex> lock(m_mutex);

    m_collecting = false;
}


std::vector<TobiiResearchGazeData> EyeTracker::getTrialSamples() const{
    std::lock_guard<std::mutex> lock(m_mutex);

    return m_trialSamples;
}
bool EyeTracker::gazeWithinError(double maxPixelMeanPosition, double maxPixelSpread, 
    FixationCoordinates fixationCoords, 
    FixationStats* fixationStats,
    int trialIndex) // we pass the trial info to write the last successful fixation window
{
    //// logging config
    //constexpr bool kLogging = true;
    //constexpr int  kLogIntervalMs = 250;   // max one log burst per this many ms

    //// throttle logging
    //static auto lastLog = std::chrono::steady_clock::time_point{};
    //const auto nowTp = std::chrono::steady_clock::now();
    //const bool shouldLog = kLogging &&   std::chrono::duration_cast<std::chrono::milliseconds>(nowTp - lastLog).count() >= kLogIntervalMs;
    //if (shouldLog) lastLog = nowTp;

    //auto log = [&](const std::string& msg) {  if (shouldLog) std::cout << "[GazeCheck] " << msg << std::endl;   };

    const int64_t windowUs = static_cast<int64_t>(m_experimentInfo.fixationTimeMs) * 1000;
    const size_t fetchCount = (static_cast<size_t>(m_experimentInfo.samplingRateHz) * m_experimentInfo.fixationTimeMs + 999) / 1000; // ceiling
    std::vector<TobiiResearchGazeData> buf;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        buf = m_samples.latest(fetchCount);
    }
    if (buf.empty()) {
        //log("FAIL: sample buffer empty");
        return false;
    }

    // reject stale data 
    int64_t now = 0;
    tobii_research_get_system_time_stamp(&now);
    const int64_t ageUs = now - buf.back().system_time_stamp;
    if (ageUs > windowUs) {
        //log(std::format("FAIL: stale data (newest sample is {:.1f} ms old, window {} ms)",   ageUs / 1000.0, m_experimentInfo.fixationTimeMs));
        return false;
    }

    // collect every sample inside the time window
    struct Pt { double x, y; };
    std::vector<Pt> leftPts, rightPts, midPts;

    const int64_t newest = buf.back().device_time_stamp;

    for (auto it = buf.rbegin(); it != buf.rend(); ++it)
    {
        if (newest - it->device_time_stamp > windowUs) break;

        const auto& l = it->left_eye.gaze_point;
        const auto& r = it->right_eye.gaze_point;

        // condition 1: every sample must be valid for both eyes
        if (l.validity != TOBII_RESEARCH_VALIDITY_VALID ||
            r.validity != TOBII_RESEARCH_VALIDITY_VALID)
        {
            //log(std::format("FAIL: invalid sample in window (left {}, right {})",   l.validity == TOBII_RESEARCH_VALIDITY_VALID ? "valid" : "INVALID",  r.validity == TOBII_RESEARCH_VALIDITY_VALID ? "valid" : "INVALID"));
            return false;
        }

        const Pt lp{ l.position_on_display_area.x * m_displayInfo.pixelWidth,  l.position_on_display_area.y * m_displayInfo.pixelHeight };
        const Pt rp{ r.position_on_display_area.x * m_displayInfo.pixelWidth,  r.position_on_display_area.y * m_displayInfo.pixelHeight };

        leftPts.push_back(lp);
        rightPts.push_back(rp);
        midPts.push_back({ (lp.x + rp.x) * 0.5, (lp.y + rp.y) * 0.5 });
    }

    // the window must contain the full expected number of samples
    const size_t totalSamplesInWindow = midPts.size();
    if (totalSamplesInWindow < fetchCount || totalSamplesInWindow < 2) {
        //log(std::format("FAIL: only {} samples in window, need {}", totalSamplesInWindow, fetchCount));
        return false;
    }

    const double n = static_cast<double>(totalSamplesInWindow);

    auto meanOf = [&](const std::vector<Pt>& pts) -> Pt {
        double mx = 0.0, my = 0.0;
        for (const auto& p : pts) {
            mx += p.x; my += p.y;
        }
        return { mx / n, my / n };
        };

    struct Dispersion { double sd; double maxDev; };
    auto dispersionOf = [&](const std::vector<Pt>& pts) -> Dispersion {
        const Pt mean = meanOf(pts);
        double sumSq = 0.0;
        double maxDevSq = 0.0;
        for (const auto& p : pts) {
            const double dx = p.x - mean.x;
            const double dy = p.y - mean.y;
            const double dSq = dx * dx + dy * dy;
            sumSq += dSq;
            maxDevSq = (std::max)(maxDevSq, dSq);
        }
        return { std::sqrt(sumSq / n), std::sqrt(maxDevSq) };
        };

    // condition 2: average L/R midpoint across all samples vs. fixation midpoint
    const Pt midMean = meanOf(midPts);
    const double fixMidX = (fixationCoords.Left.X + fixationCoords.Right.X) * 0.5;
    const double fixMidY = (fixationCoords.Left.Y + fixationCoords.Right.Y) * 0.5;

    const double meanDist = std::hypot(midMean.x - fixMidX, midMean.y - fixMidY); //distance from midpoint mean of eyes to midpoint of fixation across l/r
    const bool distOk = meanDist < maxPixelMeanPosition;

    // condition 3: no sample may deviate from its eye's mean by maxPixelSpread or more
    // this enforces that all samples are within the maxPixelSpread value of the mean value
    const Dispersion left = dispersionOf(leftPts);
    const Dispersion right = dispersionOf(rightPts);
    const bool leftOk = left.maxDev < maxPixelSpread;
    const bool rightOk = right.maxDev < maxPixelSpread;

    const bool pass = distOk && leftOk && rightOk;






    //log(std::format(
    //    "{} | n={} | midMean=({:.1f}, {:.1f}) fixMid=({:.1f}, {:.1f}) | "
    //    "dist={:.1f} / max {:.1f} [{}] | "
    //    "L: maxDev={:.1f} sd={:.1f} [{}] | R: maxDev={:.1f} sd={:.1f} [{}] | maxSpread={:.1f}",
    //    pass ? "PASS" : "FAIL", totalSamplesInWindow,
    //    midMean.x, midMean.y, fixMidX, fixMidY,
    //    meanDist, maxPixelMeanPosition, distOk ? "ok" : "TOO FAR",
    //    left.maxDev, left.sd, leftOk ? "ok" : "OUTLIER",
    //    right.maxDev, right.sd, rightOk ? "ok" : "OUTLIER",
    //    maxPixelSpread));

    if (fixationStats != nullptr)
    {
        FixationWindow* fixation = nullptr;

        if (trialIndex == 0)
            fixation = &fixationStats->fixation1;
        else if (trialIndex == 1)
            fixation = &fixationStats->fixation2;

        if (fixation != nullptr)
        {
            fixation->meanDistance = meanDist;

            fixation->spread.L = left.maxDev;
            fixation->spread.R = right.maxDev;
        }
    }

    return pass;
}