#pragma once

#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

// holds the 4 permuations per image
struct ImagePaths {
    std::string name; 
    fs::path L_orig; // <name>_L_orig.<ext>
    fs::path L_dec; // <name>_L_dec.<ext>
    fs::path R_orig; // <name>_R_orig.<ext>
    fs::path R_dec; // <name>_R_dec.<ext>
    int viewingMode; //0 = stereo   1 = left only   2 = right only
    int flickerIndex = 0; // this tracks whether the first or the second image will be flickered.
};

struct Config {
    std::string participantID;
    std::string participantAge;
    std::string participantGender;
    fs::path origImageDirectory;
    fs::path condImageDirectory;
    std::vector<ImagePaths> trials;
    fs::path outputDirectory;
    // defaults
    double flickerRate = 10.0;  // hz
    double waitTime = 2.0; // time between images
    double imageTime = 2.0; // time images are shown
    int targetFPS = 30; 
    float fovealWidth = 5.0; // degrees
    //float physicalScreenWidthMeters = 0.53; // m
    //float viewingDistanceMeters = 0.6; // m
    float pixelsPerDegree = 60.0;

    // load and parse json config file
    bool load(const std::string& configPath);

private:
    // searches the provided imageDirectory for a file named "<name><suffix>.*" (any extension).
    //returns the matched path (or empty if not found)
    fs::path findImage(const std::string& name, const std::string& suffix, const fs::path imageDirectory) const;
};
