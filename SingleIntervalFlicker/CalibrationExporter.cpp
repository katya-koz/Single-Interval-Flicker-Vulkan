//#include "CalibrationExporter.h"
//
//
//bool CalibrationExporter::init(BlockInformation blockInfo, EyeTracker eyeTracker) {
//	m_blockInfo = blockInfo;
//	m_eyetracker = eyeTracker;
//
//    openCsvLog(blockInfo);
//}
//
//bool CalibrationExporter::exportCalibrationData() {
//
//
//
//}
//
//
//void CalibrationExporter::openCsvLog(BlockInformation blockInfo) {
//
//    static const std::vector<std::string> headers = {};
//
//    m_csv = std::make_unique<CSV>();
//    m_csv->init(blockInfo, headers, false, "calibration"); // has metadata header = false, tag = "gaze"
//
//    // this csv doesn't need to function on its own thread. it will be called once, before experiment start.
//
//    
//}
