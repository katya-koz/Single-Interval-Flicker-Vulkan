#pragma once
#include <string>
#include <fstream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

class CSV {
	public:
		CSV() = default;
		~CSV();

		bool init(const std::string& participantId,
			const int participantAge,
			const char participantGender,
			//const int blockNumber, 
			const int sessionNumber,
			const int groupNumber, 
			//const int intervalMode,
			//const int displayMode,
			const std::vector<std::string>& headers,
			const std::string& outputDirectory
		); // writes headers, initializes file
		void writeRow(const std::vector<std::string>& fields); // writes data rows per response
		//void readRow(const std::)
		void close();
	private:
		std::ofstream m_file;

		// helpers
		fs::path buildPath(const std::string& participantId,/* const int blockNumber,*/ const int sessionNumber,
			const int groupNumber, const std::string& outputDir) const;
		std::string getDateString() const;
		std::string getDateTimeString() const;

};