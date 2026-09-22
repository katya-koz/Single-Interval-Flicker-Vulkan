#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Simple CSV writer used for both trial-level and gaze-level logs.
//
// NOT thread-safe on its own -- callers must make sure only one thread
// touches a given CSV instance at a time. EyeTracker funnels all of its
// gaze-log writes through a single dedicated writer thread for exactly
// this reason.
class CSV {
public:
	CSV() = default;
	~CSV();

	CSV(const CSV&) = delete;
	CSV& operator=(const CSV&) = delete;

	// `tag` is an optional extra token inserted into the filename (e.g. "gaze")
	// so multiple CSVs can share the same participant/session/block/group
	// naming without overwriting each other.
	bool init(const std::string& participantId, int participantAge, char participantGender,
		int blockNumber, int sessionNumber, int groupNumber,
		const std::vector<std::string>& headers, const std::string& outputDirectory = "",
		const std::string& tag = "");

	void writeRow(const std::vector<std::string>& fields);
	void close();

private:
	fs::path buildPath(const std::string& participantId, int blockNumber, int sessionNumber,
		int groupNumber, const std::string& outputDir, const std::string& tag) const;
	std::string getDateTimeString() const;

	std::ofstream m_file;

	// writeRow() flushes at most this often; close() always flushes
	// whatever's left buffered.
	static constexpr std::chrono::seconds m_flushInterval{ 1 };
	std::chrono::steady_clock::time_point m_lastFlush{};
};