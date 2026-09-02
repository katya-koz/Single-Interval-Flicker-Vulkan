#include "config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <Windows.h>
#include "Utils.h"

using json = nlohmann::json;

namespace {

    // Trims whitespace and stray \r (in case a CSV was saved with Windows
    // line endings and read on a system that doesn't collapse them).
    std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

    // Minimal comma splitter. Sufficient for these CSVs since none of the
    // fields contain embedded commas or quotes.
    std::vector<std::string> splitCSVLine(const std::string& line) {
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            fields.push_back(trim(field));
        }
        return fields;
    }

} // namespace

bool Config::loadConfig(const std::string& configPath) {
    // load the config.json file into app settings
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "[Config] Could not open config file: " << configPath << "\n";
        return false;
    }

    json j;
    try {
        file >> j;
    }
    catch (const json::parse_error& e) {
        std::cerr << "[Config] JSON parse error: " << e.what() << "\n";
        return false;
    }

    rootImageDirectory = j.at("Root Image Directory").get<std::string>();

    // these are technically 'optional' as they have hardcoded defaults in the config struct
    if (j.contains("Output Directory")) {
        outputDirectory = j["Output Directory"].get<std::string>();
    }
    if (j.contains("Flicker Rate (Hz)")) {
        flickerRate = j["Flicker Rate (Hz)"].get<double>();
    }
    if (j.contains("Wait Time (s)")) {
        waitTime = j["Wait Time (s)"].get<double>();
    }
    if (j.contains("Image Time (s)")) {
        imageTime = j["Image Time (s)"].get<double>();
    }
    if (j.contains("Target FPS")) {
        targetFPS = j["Target FPS"].get<int>();
    }
    if (j.contains("Display Mode")) {
        displayMode = j["Display Mode"].get<int>();
    }
    if (j.contains("Interval Mode")) {
        intervalMode = j["Interval Mode"].get<int>();
    }

    if (!fs::exists(rootImageDirectory) || !fs::is_directory(rootImageDirectory)) {
        std::string msg = "[Config] Image directory not found: " + rootImageDirectory.string();
        Utils::FatalError(msg);
        return false;
    }


    return true;
}

bool Config::loadExperimentInfo(const std::string& inputPath) {

    // expected header:
     /*
     # ============================================================                              1
     # Experiment: VESA foveation assessement                                                    2
     # Subject ID: Test                                                                          3
     # Subject Age: 20                                                                           4
     # Subject Gender: m/f                                                                       5
     # Group: 1                                                                                  6
     # Session: 1                                                                                7
     # ============================================================                              8
     imageName,Left0,Right0,Left1,Right1, posX_L, posY_L, posX_R, posY_R, order                  9
     ....
     */

    std::ifstream file(inputPath);
    if (!file.is_open()) {
        std::cerr << "[Config] Could not open trials file: " << inputPath << "\n";
        return false;
    }

    std::string line;

    while (std::getline(file, line)) {
        line = trim(line);

        // stop once we reach the main header
        if (line.starts_with("imageName,")) {
            break;
        }

        //ignore empty lines
        if (line.empty() || line[0] != '#') {
            continue;
        }

        // remove # char
        line = trim(line.substr(1));

        // ignore the ==== seperator
        if (line.empty() || line[0] == '=') {
            continue;
        }

        // split to key: value
        const size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }

        const std::string key = trim(line.substr(0, colon));
        const std::string value = trim(line.substr(colon + 1));

        if (key == "Experiment") {
            experimentInfo.experimentName = value;
        }
        else if (key == "Subject ID") {
            experimentInfo.participantID = value;
        }
        else if (key == "Subject Age") {
            experimentInfo.participantAge = std::stoi(value);
        }
        else if (key == "Subject Gender") {
            experimentInfo.participantGender = value[0];
        }
        else if (key == "Group") {
            experimentInfo.groupNumber = std::stoi(value);
        }
        else if (key == "Session") {
            experimentInfo.sessionNumber = std::stoi(value);
        }
    }

    return true;
}


// takes input file (with paths to images) and builds trial information
bool Config::loadTrials(const std::string& inputPath) {
    loadExperimentInfo(inputPath); // load header into experiment info struct

    std::ifstream file(inputPath);
    std::string line;
   

    int lineNumber = 1;
    while (std::getline(file, line)) {
        ++lineNumber;
        if (trim(line).empty() || lineNumber <= 10) continue; // skip the header block

        std::vector<std::string> fields = splitCSVLine(line);
        if (fields.size() < 10) { // ensure all header columns are present
            std::string msg = "[Config] Malformed trials.csv line " + std::to_string(lineNumber) + ": " + line;
            Utils::FatalError(msg);
            continue;
        }

        trials.push_back(parseImageRow(fields));

    }

    return true;
}

ImagePaths Config::parseImageRow(std::vector<std::string> fields) {
    ImagePaths imagePaths;

    imagePaths.name = fields[0];

    imagePaths.L_orig = rootImageDirectory / fields[1];
    imagePaths.R_orig = rootImageDirectory / fields[2];
    imagePaths.L_dec = rootImageDirectory / fields[3];
    imagePaths.R_dec = rootImageDirectory / fields[4];

    imagePaths.fixationCoords.Left.X = std::stoi(fields[5]);
    imagePaths.fixationCoords.Left.Y = std::stoi(fields[6]);
    imagePaths.fixationCoords.Right.X = std::stoi(fields[7]);
    imagePaths.fixationCoords.Right.Y = std::stoi(fields[8]);

    imagePaths.flickerIndex = std::stoi(fields[9]);

    // validate all four image paths
    if (!fs::exists(imagePaths.L_orig)) {
        Utils::FatalError("[Config] Image does not exist: " + imagePaths.L_orig.string());
    }

    if (!fs::exists(imagePaths.R_orig)) {
        Utils::FatalError( "[Config] Image does not exist: " + imagePaths.R_orig.string());
    }

    if (!fs::exists(imagePaths.L_dec)) {
        Utils::FatalError("[Config] Image does not exist: " + imagePaths.L_dec.string());
    }

    if (!fs::exists(imagePaths.R_dec)) {
        Utils::FatalError("[Config] Image does not exist: " + imagePaths.R_dec.string());
    }

    return imagePaths;
}