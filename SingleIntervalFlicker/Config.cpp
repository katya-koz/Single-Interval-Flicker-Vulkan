#include "config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <Windows.h>
#include "Utils.h"

using json = nlohmann::json;

bool Config::load(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "[Config] Could not open config file: " << configPath << "";
        return false;
    }

    json j;
    try {
        file >> j;
    }
    catch (const json::parse_error& e) {
        std::cerr << "[Config] JSON parse error: " << e.what() << "";
        return false;
    }

    participantID = j.at("Participant ID").get<std::string>();
    participantAge = std::to_string(j.at("Participant Age").get<int>());
    participantGender = j.at("Participant Gender").get<std::string>();
    origImageDirectory = j.at("Reference Image Directory").get<std::string>();
    condImageDirectory = j.at("Condition Image Directory").get<std::string>();

    std::filesystem::path p(condImageDirectory);
    conditionName = p.filename().string();

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
   
    if (j.contains("TargetFPS")) {
        targetFPS = j["TargetFPS"].get<float>();
    }

    if (!fs::exists(origImageDirectory) || !fs::is_directory(origImageDirectory)) {
        std::string msg = "[Config] Image directory not found: " + origImageDirectory.string() + "";
        Utils::FatalError(msg);
        return false;
    }

    if (!fs::exists(condImageDirectory) || !fs::is_directory(condImageDirectory)) {
        std::string msg = "[Config] Image directory not found: " + condImageDirectory.string() + "";
        Utils::FatalError(msg);
        return false;
    }

    trials.clear();

    for (const auto& trial : j.at("Trials")) {
        std::string name = trial.at("Image Name").get<std::string>();
        int viewingMode = trial.at("Viewing Mode").get<int>();

        ImagePaths img;
        img.viewingMode = viewingMode;

        img.name = name;
        img.L_orig = findImage(name, "_L", origImageDirectory);
        img.L_dec = findImage(name, "_L", condImageDirectory);
        img.R_orig = findImage(name, "_R", origImageDirectory);
        img.R_dec = findImage(name, "_R", condImageDirectory);
        std::string msg;
        // Warn about any missing permutations
        auto warn = [&](const fs::path& p, const std::string& suffix) {
            if (p.empty()){
                msg = "[Config] Warning: no file found for \"" + name + suffix + ".*\"";
                Utils::FatalError(msg);
            }
        };

        warn(img.L_orig, "_L");
        warn(img.L_dec, "_L");
        warn(img.R_orig, "_R");
        warn(img.R_dec, "_R");

        trials.push_back(img);
    }

    return true;
}

fs::path Config::findImage(const std::string& name, const std::string& suffix, const fs::path imageDirectory) const {
    std::string stem = name + suffix; // e.g. "image0_L"

    for (const auto& entry : fs::directory_iterator(imageDirectory)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();
        if (filename.rfind(stem, 0) == 0)  // starts with stem
            return entry.path();
    }

    return {};
}