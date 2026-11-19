//#pragma once
//#include <memory>
//#include <string>
//#include "csv.h"
//#include "Utils.h"
//#include "tobii_research_calibration.h"
//#include "tobii_research_eyetracker.h"
//
//class CalibrationExporter {
//public:
//    bool init(const BlockInformation& blockInfo, const DisplayInfo& displayInfo);
//
//    // Call after compute_and_apply and BEFORE freeing the result.
//    bool exportCalibrationResult(const TobiiResearchCalibrationResult* result);
//
//    // Optional: raw binary blob that can be re-applied to a tracker later.
//    bool exportBinaryCalibration(TobiiResearchEyeTracker* eyetracker);
//
//private:
//    static const char* validityToString(TobiiResearchCalibrationEyeValidity v);
//    static const char* statusToString(TobiiResearchStatus s);
//
//    BlockInformation m_blockInfo;
//    DisplayInfo m_displayInfo;
//    std::unique_ptr<CSV> m_csv;
//};