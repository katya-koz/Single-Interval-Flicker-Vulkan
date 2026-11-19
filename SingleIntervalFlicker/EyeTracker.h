#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#define DEBUG_MOUSE_GAZE
#include "csv.h"
#include "tobii_research_eyetracker.h"
#include "tobii_research_streams.h"

class EyeTracker {
public:
	EyeTracker() = default;
	~EyeTracker();

	EyeTracker(const EyeTracker&) = delete;
	EyeTracker& operator=(const EyeTracker&) = delete;

	// If participantId is non-empty, this also opens a dedicated gaze-level
	// CSV (same participant/age/gender/block/session/group naming as
	// whatever CSV you already use for trial data, tagged "gaze" so the
	// filenames don't collide) and starts a background thread that logs
	// every incoming gaze sample to it -- independent of startCollection().
	void init(float gazePrecision, float viewingDistance, const std::string& participantId = "",
		int participantAge = 0, char participantGender = 'U', int blockNumber = 0,
		int sessionNumber = 0, int groupNumber = 0, const std::string& outputDirectory = "");

	void start(TobiiResearchEyeTracker* eyetracker);
	void stop(TobiiResearchEyeTracker* eyetracker);
	bool gazeWithinError(
		int pixelError,
		int leftFixationX,
		int leftFixationY,
		int rightFixationX,
		int rightFixationY
	);
	void startCollection();
	void stopCollection();

	std::vector<TobiiResearchGazeData> getSamples();
	void clearSamples();

	bool enterCalibration(TobiiResearchEyeTracker* eyetracker);
	bool collectCalibrationPoint(TobiiResearchEyeTracker* eyetracker, float x, float y);
	bool finishCalibration(TobiiResearchEyeTracker* eyetracker);

#ifdef DEBUG_MOUSE_GAZE
	void updateMouseGaze(struct GLFWwindow* window, int monitorWidth);
#endif

private:
	static void gaze_data_callback(TobiiResearchGazeData* gaze_data, void* user_data);

	void openCsvLog(const std::string& participantId, int participantAge, char participantGender,
		int blockNumber, int sessionNumber, int groupNumber, const std::string& outputDirectory);
	void closeCsvLog();
	void writerThreadMain();
	static std::vector<std::string> gazeDataToRow(const TobiiResearchGazeData& g);

	float gazePrecisionCm = 0.f;
	float viewingDistanceCm = 0.f;

	// --- collected-sample buffer (only filled while collecting) ---
	std::mutex m_mutex;
	bool m_collecting = false;
	std::vector<TobiiResearchGazeData> m_samples;

	// --- gaze CSV log: written on its own thread so the Tobii SDK's
	// callback thread never touches the filesystem ---
	std::unique_ptr<CSV> m_csv;
	std::thread m_writerThread;
	std::atomic<bool> m_writerRunning{ false };
	std::mutex m_queueMutex;
	std::condition_variable m_queueCv;
	std::vector<TobiiResearchGazeData> m_writeQueue;
};