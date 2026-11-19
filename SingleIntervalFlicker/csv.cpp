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
    const char participantGender, const int blockNumber, const int sessionNumber,
    const int groupNumber, const std::vector<std::string>& headers,
    const std::string& outputDirectory, const std::string& tag) {

    fs::path outPath = buildPath(participantId, blockNumber, sessionNumber, groupNumber, outputDirectory, tag);

    m_file.open(outPath, std::ios::out | std::ios::trunc); // overwrite existing file of same name
    if (!m_file.is_open()) {
        Utils::FatalError("[CSV] Failed to open file: " + outPath.string());
        return false;
    }

    // metadata
    m_file << "# Age: " << participantAge << "\n";
    m_file << "# Gender: " << participantGender << "\n";
    m_file << "# Timestamp: " << getDateTimeString() << "\n";

    // column headers
    for (size_t i = 0; i < headers.size(); i++) {
        m_file << headers[i];
        if (i < headers.size() - 1) m_file << ",";
    }
    m_file << "\n";
    m_file.flush();
    m_lastFlush = std::chrono::steady_clock::now();

    std::cout << "[CSV] Opened: " << outPath.string() << "\n";
    return true;
}

void CSV::writeRow(const std::vector<std::string>& fields) {
    if (!m_file.is_open()) return;

    for (size_t i = 0; i < fields.size(); i++) {
        // quote any field that contains a comma.. just for safety
        if (fields[i].find(',') != std::string::npos)
            m_file << "\"" << fields[i] << "\"";
        else
            m_file << fields[i];

        if (i < fields.size() - 1) m_file << ",";
    }
    m_file << "\n";

    // Flush at most once a second -- flushing every row is unnecessary
    // overhead at sampling rates like 120 Hz and can bottleneck the thread
    // doing the writing. Worst case on an unclean shutdown you lose the
    // last <1s of buffered rows; close() below flushes any remainder on a
    // clean shutdown.
    auto now = std::chrono::steady_clock::now();
    if (now - m_lastFlush >= m_flushInterval) {
        m_file.flush();
        m_lastFlush = now;
    }
}

void CSV::close() {
    if (m_file.is_open()) {
        m_file.flush();
        m_file.close();
    }
}

fs::path CSV::buildPath(const std::string& participantId, const int blockNumber, const int sessionNumber,
    const int groupNumber, const std::string& outputDir, const std::string& tag) const {

    fs::path dir = outputDir.empty() ? fs::current_path() : fs::path(outputDir);

    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }

    std::string base =
        "G" + std::to_string(groupNumber) + "_" +
        participantId + "_" +
        "S" + std::to_string(sessionNumber) + "_" +
        "B" + std::to_string(blockNumber) +
        (tag.empty() ? "" : ("_" + tag)) +
        ".csv";

    return dir / base;
}

std::string CSV::getDateTimeString() const {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm tm;

    localtime_s(&tm, &t);

    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}