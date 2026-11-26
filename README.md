# Single Interval Flicker Experiment

This project is the implementation of an ISO Flicker Paradigm experiment using Vulkan integrated with Tobii Eye Tracker (Fusion Pro) @120Hz. The render engine is suited for a Wheatstone stereoscope set up. Custom stereoscopic calibration (records calibration points, basic GUI) software is available here: https://github.com/katya-koz/stereoscopic-calibration.

## Installation & Setup Instructions

1. Install [VulkanSDK](https://vulkan.lunarg.com/sdk/home)
   - _Ensure that the 'Shader Toolchain Debug Symbols - 64-bit' option is checked under the 'Select Components' step of the installation._
2. Install [OpenCV](https://opencv.org/releases/)
3. Install [GLFW](https://www.glfw.org/download.html)
4. Update environment variables as follows:

| Variable    | Value                                                   |
| ----------- | ------------------------------------------------------- |
| OpenCV_Path | [Your path to openCV here: eg C:\opencv\build]          |
| GLFW_Path   | [Your path to GLFW here: eg C:\glfw\glfw-3.4.bin.WIN64] |
| Vulkan_Path | [Your path to Vulkan here: eg C:\VulkanSDK\1.4.341.1]   |

## Running + Config

- To run: ` ./SingleIntervalFlicker.exe trials.csv [--config <config.json>]
  - No --config flag: runs default config location (located in x64 release folder)

- Example output (output folder as described in config.json):

```
# Age: 22
# Gender: F
# Timestamp: 2026-11-23_14-57-20
Trial Index,Image ID,Codec,Foveation,Image,Interval,Position-X (Left),Position-Y (Left),Position-X (Right),Position-Y (Right),Mode,Response,Duration,Subject,Fixation Window (ms),Max Mean Position (pixels),Max Standard Deviation (pixels)
0,0,null,I,Desk,1,2232,638,2292,638,Stereo,1,29814,testSubject,150,74.319372,29.723787
1,1,null,N,Creek,1,984,1080,1084,1080,Stereo,0,1201,testSubject,150,74.319372,29.723787
2,2,null,F,Creek,0,984,1080,1084,1080,Stereo,0,1167,testSubject,150,74.319372,29.723787
...
```

- Example gaze output (if eye tracking is on)
  - If max fixation period is exceeded, trial will be skipped and re-done at end of block. Unfinished/skipped trials fill in missing values with '-1'

```
system_time_stamp,device_time_stamp,left_gaze_x,left_gaze_y,left_gaze_valid,left_pupil_diameter,left_pupil_valid,right_gaze_x,right_gaze_y,right_gaze_valid,right_pupil_diameter,right_pupil_valid,current_stage,current_stage_name,trial index, experiment_time_s,event_number
3642233240837,3642290835095,0.500929475,0.417856008,1,4.14570618,1,0.524051607,0.435305208,1,4.36961365,1,1,Fixation,0,2.665262,0
3642233249170,3642290843428,0.526468337,0.35570097,1,4.17668152,1,0.549048841,0.385969967,1,4.42718506,1,1,Fixation,0,2.673595,0
...

```

- The graphics pipeline is only configured to work with PPM images (for now...)
- This experiment uses one window for both monitors. Thus, it is important that the display orientation is correct in settings.
  - The **primary monitor** will be considered the left most monitor. Please ensure that the Windows display settings reflect this:

```
   ______________________      ______________________
  |                    |      |                    |
  |                    |      |                    |
  | 1.                 |      | 2.                 |
  ______________________      ______________________
Left monitor                  Right monitor
```

### Prepare a configuration file (i.e. config.json):

```
{
  "Root Image Directory": "D:\\FoveationAssessment\\images\\",
  "Output Directory": "D:\\FoveationAssessment\\Data",
  "Flicker Rate (Hz)": 5,
  "Wait Time (s)": 1,
  "Image Time (s)": 0.6,
  "Display Mode": 0,
  "Interval Mode": 0,
  "Gaze Precision Center (degrees)": 2.5,
  "Gaze Precision Edge (degrees)": 2.5,
  "Maximum Spread (degrees)": 1.0,
  "Fixation Time (ms)": 150,
  "Max Fixation Time (s)": 10.0,
  "Viewing Distance (cm)": 62.0,
  "Eye Tracking": true
}
```

| Configuration Item              | Expected Type / Values        | Description                                                                                                                       |
| ------------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Root Image Directory            | `String` (directory path)     | Directory from which the program reads stimulus images.                                                                           |
| Output Directory                | `String` (directory path)     | Directory where output data files are saved.                                                                                      |
| Flicker Rate (Hz)               | `Integer` or `Float`          | Rate at which images flicker, measured in Hertz.                                                                                  |
| Wait Time (s)                   | `Integer` or `Float`          | Minimum time spent on the fixation screen before the fixation timer begins. Can be set to `0`.                                    |
| Image Time (s)                  | `Integer` or `Float`          | Total duration for which the stimulus is displayed.                                                                               |
| Display Mode                    | `Integer` (`0` or `1`)        | Selects SDR/HDR display mode. Currently not applicable; leave at `0` for SDR.                                                     |
| Interval Mode                   | `Integer` (`0` or `1`)        | Selects single- or two-interval mode. Currently not applicable; leave at `0` for two-interval mode.                               |
| Gaze Precision Center (degrees) | `Integer` or `Float`          | Required gaze precision at the center of the screen, measured in degrees. Represents the DIAMETER of the error boundary.          |
| Gaze Precision Edge (degrees)   | `Integer` or `Float`          | Required gaze precision at the farthest corner of the screen, measured in degrees. Represents the DIAMETER of the error boundary. |
| Maximum Spread (degrees)        | `Integer` or `Float`          | Maximum allowable gaze spread per eye for fixation, measured in degrees.                                                          |
| Fixation Time (ms)              | `Integer`                     | Duration for which fixation must be maintained before proceeding to the stimulus.                                                 |
| Max Fixation Time (s)           | `Integer` or `Float`          | Maximum time allowed to achieve fixation before skipping and retrying the trial. Set to `0` to disable retrialing.                |
| Viewing Distance (cm)           | `Integer` or `Float`          | Distance between the participant's eyes and the display screen, measured in centimeters.                                          |
| Eye Tracking                    | `Boolean` (`true` or `false`) | Enables (`true`) or disables (`false`) eye tracking.                                                                              |

### Prepare a trials.csv file

// to be continued...

## Experiment

- This program assumes an experimental setup with two identical monitors.
- Left and right arrow keys are used to answer, or gamepad's L and R triggers.

## Fixation Information

- During the fixation period, poll fixation status on every game loop (~10 ms), using the most recent ⌊samplingRateHz × fixationTimeMs / 1000⌋ samples.
- When eye tracking is enabled and the system is in the fixation period, all of the following conditions must be satisfied for the fixation window to proceed:
  1. Sample validity: All samples in the fixation window must be valid.
  2. Gaze spread: No gaze point for either eye may fall outside the accepted Spread threshold.
  3. Mean position drift: Calculate the mean gaze position for each eye across all samples, then calculate the midpoint between the two means. This midpoint must be within the maximum allowed mean position drift of the fixation target.

---

`This program was created by Ekaterina (Katya) Kozlovsky under The Centre for Vision Research at York University, Toronto, Canada.`
