#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include "RenderTypes.h"
#include "DisplayInfo.h"
#include "FixationCoords.h"


namespace fs = std::filesystem;

struct ExperimentInformation {
    double flickerRate = 10.0;  // hz
    double waitTime = 0.6; // minimum time between images
    double imageTime = 2.0; // time images are shown
    int fixationTimeMs; // time particpant should be fixated before showing image
    double maxFixationTimeS = 0.0; // maximum amount of time for participant ot be in the fixation stage before forcing to proceed; 0 or any other value < fixationtimeMs flags to not apply a max fixation time
    int intervalMode = 1; // 0 = two interval; 1 = single interval (two images, side by side)
    int displayMode = 1; // 0 = SDR only ; 1 = HDR preferred
    double gazePrecisionCenterDegrees = 1.0;
    double gazePrecisionEdgeDegrees = gazePrecisionCenterDegrees;
    double maxSpreadDegrees = 2.5;
    double viewingDistanceCm = 60.0;
    bool isEyeTracking = true;
    int samplingRateHz = 120; // this one isnt configurable, yet
};


// holds the 4 permuations per image and  
// all information for current trial including max spread,
// fixation coords and max mean position
struct TrialInformation {
    FixationCoordinates fixationCoords = FixationCoordinates();
    std::string name;
    int index = -1; // this is used to keep track of the order so that if a trial is repeated, we know which one. 
    std::string codec = "N/a";
    std::string foveatLevel = "N/a";

    fs::path L_orig; // <name>_L_orig.<ext>
    fs::path L_dec; // <name>_L_dec.<ext>
    fs::path R_orig; // <name>_R_orig.<ext>
    fs::path R_dec; // <name>_R_dec.<ext>

    int imageWidth = 3840;
    int imageHeight = 2160;
    int viewingMode = 0; //0 = stereo   1 = left only   2 = right only;; default is Stereo
    int flickerIndex = 0; // this tracks whether the first or the second image will be flickered. Populated from the 'order' column in input csv.
    bool isControl = false;

    double maxMeanPositionPixels;
    double maxSpreadPixels;

   
};

struct BlockInformation {
    std::string experimentName = "UND";
    std::string participantID = "UND";
    int block = 0;
    int participantAge = 0;
    char participantGender = 'U';
    int groupNumber = 0;
    int sessionNumber = 0;
    fs::path outputDirectory = "C://flickerTestOutput"; // where the results csv is output
};

struct Config {
    fs::path rootImageDirectory; 
    std::vector<TrialInformation> trials;
       
    ExperimentInformation experimentInfo = ExperimentInformation();
    BlockInformation blockInfo = BlockInformation();


    // load and parse the json config
    bool loadConfig(const std::string& configPath);

    // loads experiment info such as participant id, age, gender, session #, group #
    bool loadExperimentInfo(const std::string& inputPath);

    // load and parse the trial ordercsv  into `trials`
    bool loadTrials(const std::string& inputPath, DisplayInfo displayInfo);


    std::string toString() const;
    private:
        // helper to read csv row and load into TrialInformation, to add to 'trials'
        TrialInformation parseImageRow(const std::vector<std::string> fields, DisplayInfo displayInfo, int index);
   


};


