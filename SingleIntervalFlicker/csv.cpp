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

bool CSV::init(const std::string& participantId, const std::string& participantAge, const std::string& participantGender, const std::string& experimentName, const std::string& variant, const std::vector<std::string>& headers, const std::string& outputDirectory = "") {
    
    fs::path outPath = buildPath(participantId, experimentName, variant, outputDirectory);

    m_file.open(outPath);
    if (!m_file.is_open()) {
        Utils::FatalError("[CSV] Failed to open file: " + outPath.string());
        return false;
    }

    // metadata
    m_file << "Experiment: " << experimentName << "\n";
    m_file << "Variant: " << variant << "\n";
    m_file << "Participant ID: " << participantId << "\n";
    m_file << "Participant Age: " << participantAge << "\n";
    m_file << "Participant Gender: " << participantGender << "\n";
    m_file << "Start Time: " << getDateTimeString() << "\n";
    m_file << "\n";

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
        // quote any field that contains a comma
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

fs::path CSV::buildPath(const std::string& participantId, const std::string& experimentName, const std::string& variant, const std::string& outputDir) const {
    fs::path dir = outputDir.empty() ? fs::current_path() : fs::path(outputDir);

    // create the directory if it doesn't exist
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
    // sanitize for file names
    std::string sanitizedExperimentName = experimentName;
    std::replace(sanitizedExperimentName.begin(), sanitizedExperimentName.end(), ' ', '-');

    std::string sanitizedVariantName = variant;
    std::replace(sanitizedVariantName.begin(), sanitizedVariantName.end(), ' ', '-');

    std::string base = sanitizedExperimentName + "_" + sanitizedVariantName + "_" + participantId + "_" + getDateString();
    int counter = 0;
    fs::path outPath;

    do {
        outPath = dir / (base + "_" + std::to_string(counter) + ".csv");
        counter++;
    } while (fs::exists(outPath));

    return outPath;
}

std::string CSV::getDateString() const {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm tm;

    localtime_s(&tm, &t);

    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d");
    return ss.str();
}

std::string CSV::getDateTimeString() const {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm tm;

    localtime_s(&tm, &t);

    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}