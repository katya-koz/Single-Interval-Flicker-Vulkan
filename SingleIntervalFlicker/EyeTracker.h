#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
//#define DEBUG_MOUSE_GAZE
#include "csv.h"
#include "tobii_research_eyetracker.h"
#include "tobii_research_streams.h"

#include "RingBuffer.h"
#include <cstdint>
#include <limits>
#include "DisplayInfo.h"
#include "Config.h"
#include "TrialPhase.h"


struct GazeOffset {
	float lx = 0, ly = 0, rx = 0, ry = 0;   // measured - target, in pixels
	bool lValid = false, rValid = false;
};


struct FixationStats { // stats for fixation window 
	int timeWindowMS;
	float meanPositionDistance_0_L;
	float meanPositionDistance_0_R;
	float meanPositionDistance_1_L;
	float meanPositionDistance_1_R;
	float stdDev_0_L;
	float stdDev_0_R;
	float stdDev_1_L;
	float stdDev_1_R;
};

class EyeTracker {

	

public:
	EyeTracker() = default;
	~EyeTracker();

	EyeTracker( EyeTracker&) = delete;
	EyeTracker& operator=( EyeTracker&) = delete;


	void init( ExperimentInformation experimentInfo, DisplayInfo displayInfo, BlockInformation blockInfo);
	void start(TobiiResearchEyeTracker* eyetracker);
	void stop(TobiiResearchEyeTracker* eyetracker);

	void startCollection();
	void stopCollection();

	DisplayInfo m_displayInfo; 
	std::vector<TobiiResearchGazeData> getTrialSamples() const;
	void clearSamples();

	static constexpr size_t kGazeHz = 120;
	static constexpr size_t kFixationWindowSec = 3; // this is somewhat arbirtrary, but 3 s is plenty of time to fixate
	static constexpr size_t kSampleCapacity = kGazeHz * kFixationWindowSec;

	void markExperimentStart();   // defines t = 0 for the gaze log

	bool gazeWithinError(double maxPixelMeanPosition, double maxPixelSpread, FixationCoordinates fixationCoords);
	bool getLatestSample(TobiiResearchGazeData& out) const;

	void updateStage(TrialPhase mode, int trialIndex);
	void updateEventNumber(int eventNumber);


private:
	int m_fixationTimeMs;
	static constexpr int64_t kOriginUnset = (std::numeric_limits<int64_t>::min)();
	std::atomic<int64_t> m_originUs{ kOriginUnset };
	std::vector<std::string> gazeDataToRow(const TobiiResearchGazeData& g) const; // no longer static

	static void gaze_data_callback(TobiiResearchGazeData* gaze_data, void* user_data);

	void openCsvLog(BlockInformation blockInfo);
	void closeCsvLog();
	void writerThreadMain();

	TrialPhase currentStage;
	int currentTrialIndex;
	int currentEventNumber = -1; 

	ExperimentInformation m_experimentInfo; 

	//collected-sample buffer (only filled while collecting)
	mutable std::mutex m_mutex;
	bool m_collecting = false;
	RingBuffer<TobiiResearchGazeData> m_samples{ kSampleCapacity };
	std::vector<TobiiResearchGazeData> m_trialSamples;

	std::unique_ptr<CSV> m_csv;
	std::thread m_writerThread;
	std::atomic<bool> m_writerRunning{ false };
	std::mutex m_queueMutex;
	std::condition_variable m_queueCv;
	std::vector<TobiiResearchGazeData> m_writeQueue;
};