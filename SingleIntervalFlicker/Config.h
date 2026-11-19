#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include "RenderTypes.h"

namespace fs = std::filesystem;

// holds the 4 permuations per image
struct ImagePaths {
    FixationCoordinates fixationCoords = FixationCoordinates();
    std::string name;
    std::string codec = "N/a"; 
    std::string foveatLevel = "N/a";
    fs::path L_orig; // <name>_L_orig.<ext>
    fs::path L_dec; // <name>_L_dec.<ext>
    fs::path R_orig; // <name>_R_orig.<ext>
    fs::path R_dec; // <name>_R_dec.<ext>
    int imageWidth = 3840; // maybe a little lazy... but this is hardcoded for now so that i dont spend all night. to be fixed later.. 
    int imageHeight = 2160;
    int viewingMode = 0; //0 = stereo   1 = left only   2 = right only;; default is Stereo
    int flickerIndex = 0; // this tracks whether the first or the second image will be flickered. Populated from the 'order' column in input csv.
};

struct ExperimentInformation {
    std::string experimentName;
    std::string participantID;
    int block;
    int participantAge;
    char participantGender; 
    int groupNumber;
    int sessionNumber;
};

struct Config {
    fs::path rootImageDirectory; 
    std::vector<ImagePaths> trials;
    fs::path outputDirectory = "C://flickerTestOutput"; // where the results csv is output
    int intervalMode = 1; // 0 = two interval; 1 = single interval (two images, side by side)
    int displayMode = 1; // 0 = SDR only ; 1 = HDR preferred
    float gazePrecisionDegrees = 1.0;
    float viewingDistanceCm = 60.0;
    ExperimentInformation experimentInfo = ExperimentInformation();

    // defaults
    double flickerRate = 10.0;  // hz
    double waitTime = 0.6; // time between images
    double imageTime = 2.0; // time images are shown
    //int targetFPS = 30; // removed. frame lock interferes with vulkan's natural v sync and messes up the timings for scene switching.

    // load and parse the json config
    bool loadConfig(const std::string& configPath);

    // loads experiment info such as participant id, age, gender, session #, group #
    bool loadExperimentInfo(const std::string& inputPath);

    // load and parse the trial ordercsv  into `trials`
    bool loadTrials(const std::string& inputPath);

    // helper to read csv row and load into ImagePaths, to add to 'trials'
    
    ImagePaths parseImageRow(const std::vector<std::string> fields);

};


