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

    // Trims whitespace and stray \r 
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
        blockInfo.outputDirectory = j["Output Directory"].get<std::string>();
    }
    if (j.contains("Flicker Rate (Hz)")) {
        experimentInfo.flickerRate = j["Flicker Rate (Hz)"].get<double>();
    }
    if (j.contains("Wait Time (s)")) {
        experimentInfo.waitTime = j["Wait Time (s)"].get<double>();
    }
    if (j.contains("Image Time (s)")) {
        experimentInfo.imageTime = j["Image Time (s)"].get<double>();
    }
    if (j.contains("Gaze Precision Center (degrees)")) {
        experimentInfo.gazePrecisionCenterDegrees = j["Gaze Precision Center (degrees)"].get<double>();
    }
    if (j.contains("Gaze Precision Edge (degrees)")) {
        experimentInfo.gazePrecisionEdgeDegrees = j["Gaze Precision Edge (degrees)"].get<double>();
    }
    if (j.contains("Viewing Distance (cm)")) {
        experimentInfo.viewingDistanceCm = j["Viewing Distance (cm)"].get<double>();
    }
    if (j.contains("Display Mode")) {
        experimentInfo.displayMode = j["Display Mode"].get<int>();
    }
    if (j.contains("Interval Mode")) {
        experimentInfo.intervalMode = j["Interval Mode"].get<int>();
    }
    if (j.contains("Eye Tracking")) {
        experimentInfo.isEyeTracking = j["Eye Tracking"].get<bool>();
    }
    if (j.contains("Maximum Spread (degrees)")) {
        experimentInfo.maxSpreadDegrees = j["Maximum Spread (degrees)"].get<double>();
    }
    if (j.contains("Fixation Time (ms)")) {
        experimentInfo.fixationTimeMs = j["Fixation Time (ms)"].get<double>();
    }
    if (j.contains("Max Fixation Time (s)")) {
        experimentInfo.maxFixationTimeS = j["Max Fixation Time (s)"].get<int>();
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
     # ============================================================                                                                        1
     # Experiment: VESA foveation assessement                                                                                              2
     # Subject ID: Test                                                                                                                    3
     # Subject Age: 20                                                                                                                     4
     # Subject Gender: m/f                                                                                                                 5
     # Group: 1                                                                                                                            6
     # Session: 1                                                                                                                          7
     # Block: 1                                                                                                                            8
     # ============================================================                                                                        9
     codec,imageName, foveatLevels, leftImage0,rightImage0,leftImage1,rightImage1, posX_L, posY_L, posX_R, posY_R, order                   10
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
            blockInfo.experimentName = value;
        }
        else if (key == "Subject ID") {
            blockInfo.participantID = value;
        }
        else if (key == "Subject Age") {
            blockInfo.participantAge = std::stoi(value);
        }
        else if (key == "Subject Gender") {
            blockInfo.participantGender = value[0];
        }
        else if (key == "Group") {
            blockInfo.groupNumber = std::stoi(value);
        }
        else if (key == "Session") {
            blockInfo.sessionNumber = std::stoi(value);
        }
        else if (key == "Block") {
            blockInfo.block = std::stoi(value);
        }
    }

    return true;
}


// takes input file (with paths to images) and builds trial information
bool Config::loadTrials(const std::string& inputPath, DisplayInfo displayInfo) {
    loadExperimentInfo(inputPath); // load header into experiment info struct

    std::ifstream file(inputPath);
    std::string line;
   
    int idx = 0;
    int lineNumber = 1;
    while (std::getline(file, line)) {
        ++lineNumber;
        if (trim(line).empty() || lineNumber <= 11) continue; // skip the header block
       
        std::vector<std::string> fields = splitCSVLine(line);
        if (fields.size() < 10) { // ensure all header columns are present
            std::string msg = "[Config] Malformed trials.csv line " + std::to_string(lineNumber) + ": " + line;
            Utils::FatalError(msg);
            continue;
        }

        trials.push_back(parseImageRow(fields, displayInfo, idx));
        idx++;
    }

    return true;
}

TrialInformation Config::parseImageRow(std::vector<std::string> fields, DisplayInfo displayInfo, int index) {
    TrialInformation trialInfo;

    trialInfo.index = index;
    trialInfo.name = fields[1];
    trialInfo.codec = fields[0];
    trialInfo.foveatLevel = fields[2];
    trialInfo.L_orig = rootImageDirectory / fields[2] / fields[3];
    trialInfo.R_orig = rootImageDirectory / fields[2] / fields[4];
    trialInfo.L_dec = rootImageDirectory / fields[2] / fields[5];
    trialInfo.R_dec = rootImageDirectory / fields[2] / fields[6];

    trialInfo.fixationCoords.Left.X = std::stoi(fields[7]);
    trialInfo.fixationCoords.Left.Y = std::stoi(fields[8]);
    trialInfo.fixationCoords.Right.X = std::stoi(fields[9]);
    trialInfo.fixationCoords.Right.Y = std::stoi(fields[10]);

    trialInfo.flickerIndex = std::stoi(fields[11]);


    if (Utils::containsControl(trialInfo.foveatLevel) || Utils::containsControl(trialInfo.name)) {
        trialInfo.isControl = true;

    }

    if (trialInfo.flickerIndex != 0 && trialInfo.flickerIndex != 1) {
        std::string msg = "[Config] Error in 'order' value in trial csv: Must be one of: 0, 1. Is: 2";
        Utils::FatalError(msg);
    }
    // calculate max mean position distance and max spread
    Coords centerCoordinates = { displayInfo.pixelWidth / 4,  displayInfo.pixelHeight / 2 };

    trialInfo.maxMeanPositionPixels = 
            Utils::degreesToPixelError (
                experimentInfo.gazePrecisionCenterDegrees,
                experimentInfo.gazePrecisionEdgeDegrees,
                trialInfo.fixationCoords,
                centerCoordinates,
                displayInfo
            );

    trialInfo.maxSpreadPixels = Utils::degreesToPixel(  experimentInfo.maxSpreadDegrees, displayInfo );

    // validate all four image paths
    if (!fs::exists(trialInfo.L_orig)) {
        Utils::FatalError("[Config] Image does not exist: " + trialInfo.L_orig.string());
    }

    if (!fs::exists(trialInfo.R_orig)) {
        Utils::FatalError( "[Config] Image does not exist: " + trialInfo.R_orig.string());
    }

    if (!fs::exists(trialInfo.L_dec)) {
        Utils::FatalError("[Config] Image does not exist: " + trialInfo.L_dec.string());
    }

    if (!fs::exists(trialInfo.R_dec)) {
        Utils::FatalError("[Config] Image does not exist: " + trialInfo.R_dec.string());
    }

    return trialInfo;
}


#include <sstream>
#include <iomanip>

std::string Config::toString() const
{
    std::ostringstream out;

    out << std::boolalpha << std::fixed << std::setprecision(3);

    out << "========== CONFIG DEBUG DUMP ==========\n\n";

    // Root configuration
    out << "[Config]\n";
    out << "rootImageDirectory: " << rootImageDirectory.string() << '\n';
    out << "trials.size(): " << trials.size() << "\n\n";

    // ExperimentInformation
    out << "[ExperimentInformation]\n";
    out << "flickerRate: " << experimentInfo.flickerRate << '\n';
    out << "waitTime: " << experimentInfo.waitTime << '\n';
    out << "imageTime: " << experimentInfo.imageTime << '\n';
    out << "fixationTimeMs: " << experimentInfo.fixationTimeMs << '\n';
    out << "intervalMode: " << experimentInfo.intervalMode << '\n';
    out << "displayMode: " << experimentInfo.displayMode << '\n';
    out << "gazePrecisionCenterDegrees: "  << experimentInfo.gazePrecisionCenterDegrees << '\n';
    out << "gazePrecisionEdgeDegrees: "  << experimentInfo.gazePrecisionEdgeDegrees << '\n';
    out << "maxSpreadDegrees: " << experimentInfo.maxSpreadDegrees << '\n';
    out << "viewingDistanceCm: " << experimentInfo.viewingDistanceCm << '\n';
    out << "isEyeTracking: " << experimentInfo.isEyeTracking << '\n';
    out << "maxFixationTime(s): " << experimentInfo.maxFixationTimeS << "\n";
    out << "samplingRateHz: " << experimentInfo.samplingRateHz << "\n\n";
    out << "isMaxFixation: " << (experimentInfo.fixationTimeMs < experimentInfo.maxFixationTimeS * 1000 && experimentInfo.isEyeTracking);

    //// BlockInformation
    //out << "[BlockInformation]\n";
    //out << "experimentName: " << blockInfo.experimentName << '\n';
    //out << "participantID: " << blockInfo.participantID << '\n';
    //out << "block: " << blockInfo.block << '\n';
    //out << "participantAge: " << blockInfo.participantAge << '\n';
    //out << "participantGender: " << blockInfo.participantGender << '\n';
    //out << "groupNumber: " << blockInfo.groupNumber << '\n';
    //out << "sessionNumber: " << blockInfo.sessionNumber << '\n';
    //out << "outputDirectory: "
    //    << blockInfo.outputDirectory.string() << "\n\n";

    //// TrialInformation vector
    //out << "========== TRIALS (" << trials.size() << ") ==========\n";

    //for (std::size_t i = 0; i < trials.size(); ++i)
    //{
    //    const TrialInformation& trial = trials[i];

    //    out << "\n--- Trial " << i << " ---\n";

    //    out << "name: " << trial.name << '\n';
    //    out << "codec: " << trial.codec << '\n';
    //    out << "foveatLevel: " << trial.foveatLevel << '\n';

    //    out << "L_orig: " << trial.L_orig.string() << '\n';
    //    out << "L_dec: " << trial.L_dec.string() << '\n';
    //    out << "R_orig: " << trial.R_orig.string() << '\n';
    //    out << "R_dec: " << trial.R_dec.string() << '\n';

    //    out << "imageWidth: " << trial.imageWidth << '\n';
    //    out << "imageHeight: " << trial.imageHeight << '\n';
    //    out << "viewingMode: " << trial.viewingMode << '\n';
    //    out << "flickerIndex: " << trial.flickerIndex << '\n';

    //    out << "maxMeanPositionPixels: "
    //        << trial.maxMeanPositionPixels << '\n';
    //    out << "maxSpreadPixels: "
    //        << trial.maxSpreadPixels << '\n';

    //    // Fixation coordinates
    //    out << "fixationCoords:\n";
    //    out << "  Left.X: " << trial.fixationCoords.Left.X << '\n';
    //    out << "  Left.Y: " << trial.fixationCoords.Left.Y << '\n';
    //    out << "  Right.X: " << trial.fixationCoords.Right.X << '\n';
    //    out << "  Right.Y: " << trial.fixationCoords.Right.Y << '\n';
    //}

    //out << "\n========== END CONFIG DEBUG DUMP ==========\n";

    return out.str();
}