#include "csv.h"
#include "Utils.h"
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <iostream>

CSV::~CSV() {
    close();
}

bool CSV::init(const std::string& participantId, const int participantAge,
    const char participantGender, /*const int blockNumber,*/ const int sessionNumber,
    const int groupNumber,/*const int intervalMode, const int displayMode,*/
    const std::vector<std::string>& headers, const std::string& outputDirectory ="") {
    
    fs::path outPath = buildPath(participantId, /*blockNumber,*/ sessionNumber, groupNumber, outputDirectory);

    m_file.open(outPath);
    if (!m_file.is_open()) {
        Utils::FatalError("[CSV] Failed to open file: " + outPath.string());
        return false;
    }

    //std::string intervalModeString = intervalMode == 0 ? "two-interval" : "single-interval";
    //std::string displayModeString = displayMode == 0 ? "SDR" : "HDR";

    // metadata
    m_file << "# Age: " << participantAge << "\n";
    m_file << "# Gender: " << participantGender << "\n";
   

    // column headers
    for (int i = 0; i < headers.size(); i++) {
        m_file << headers[i];
        if (i < headers.size() - 1) m_file << ",";
    }
    m_file << "\n";
    m_file.flush();

    std::cout << "[CSV] Opened: " << outPath.string() << "\n";
    return true;
}

void CSV::writeRow(const std::vector<std::string>& fields) {
    if (!m_file.is_open()) return;

    for (int i = 0; i < fields.size(); i++) {
        // quote any field that contains a comma.. just for safety
        if (fields[i].find(',') != std::string::npos)
            m_file << "\"" << fields[i] << "\"";
        else
            m_file << fields[i];

        if (i < fields.size() - 1) m_file << ",";
    }
    m_file << "\n";
    m_file.flush();
}

void CSV::close() {
    if (m_file.is_open()) m_file.close();
}

fs::path CSV::buildPath(const std::string& participantId, /* const int blockNumber,*/ const int sessionNumber,
    const int groupNumber, const std::string& outputDir) const {
    fs::path dir = outputDir.empty() ? fs::current_path() : fs::path(outputDir);

    // create the directory if it doesn't exist
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }


    std::string base = "G" + std::to_string(groupNumber) + "_" + participantId + "_" + "S" + std::to_string(sessionNumber) + "_" + getDateTimeString() + ".csv"; // base for counting block numbers

     

    //int blockNumberCounter = 1;
    fs::path outPath;

    do {
        //outPath = dir / (base + "_" + "B" + std::to_string(blockNumberCounter) + "_" + getDateTimeString() + ".csv");
        outPath = dir / base;
        //blockNumberCounter++;
    } while (fs::exists(outPath));

    return outPath;
}
// not currently needed
//std::string CSV::getDateString() const {
//    auto now = std::chrono::system_clock::now();
//    std::time_t t = std::chrono::system_clock::to_time_t(now);
//
//    std::tm tm;
//
//    localtime_s(&tm, &t);
//
//    std::ostringstream ss;
//    ss << std::put_time(&tm, "%Y-%m-%d");
//    return ss.str();
//}
//
std::string CSV::getDateTimeString() const {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm tm;

    localtime_s(&tm, &t);

    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}