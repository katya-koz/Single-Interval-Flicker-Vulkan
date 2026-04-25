# Single Interval Flicker Experiment

This project is the implementation of an ISO Flicker Paradigm experiment using Vulkan. 

## Installation & Setup Instructions
1. Install [VulkanSDK](https://vulkan.lunarg.com/sdk/home)
2. Install [OpenCV](https://opencv.org/releases/)
3. Install [GLFW](https://www.glfw.org/download.html)
4. Update environment variables as follows:
   
| Variable      | Value |
| ----------- | ----------- |
| OpenCV_Path      | [Your path to openCV here: eg C:\opencv\build]      |
| GLFW_Path   | [Your path to GLFW here: eg C:\glfw\glfw-3.4.bin.WIN64]        |
| Vulkan_Path   | [Your path to Vulkan here: eg C:\VulkanSDK\1.4.341.1] |

## Running + Config
* Ensure that both monitors (are HDR compatible, and) are set to HDR mode in Windows (there will be an error otherwise)
* To run: `SingleIntervalFlicker.exe [config here]`
* The graphics pipeline is only configured to work with PPM images (for now...)
* This experiment uses one window for both monitors. Thus, it is important that the display orientation is correct in settings.
  * The **primary monitor** will be considered the left most monitor. Please ensure that the Windows display settings reflect this:
```
   ______________________      ______________________ 
  |                    |      |                    |
  |                    |      |                    |
  | 1.                 |      | 2.                 |
  ______________________      ______________________
Left monitor                  Right monitor
 ```
* Identify the image directory and names in the config file. Each image should have 2 permuations per folder, ie `image0_L.ppm` and `image0_R.ppm`.
Here is an example configuration file (Viewing mode: 0 = stereo, 1 = left only, 2 = right only).
```
{
  "Participant ID": "TestID",
  "Participant Age": 30,
  "Participant Gender": "F",
  "Reference Image Directory": "C:\\PPM\\orig",
  "Condition Image Directory": "C:\\PPM\\jpeg",
  "Output Directory": "C:\\flickerOutput",
  "TargetFPS": 30,
  "Flicker Rate (Hz)": 5,
  "Wait Time (s)": 2,
  "Image Time (s)": 8,
  "Trials": [
    {
      "Image Name": "image0",
      "Viewing Mode": 0
    },
    {
      "Image Name": "image1",
      "Viewing Mode": 0
    },
    {
      "Image Name": "image2",
      "Viewing Mode": 0
    },
    {
      "Image Name": "image3",
      "Viewing Mode": 0
    },
    {
      "Image Name": "image4",
      "Viewing Mode": 0
    },
    {
      "Image Name": "image5",
      "Viewing Mode": 0
    },
    {
      "Image Name": "image6",
      "Viewing Mode": 0
    },
    {
      "Image Name": "image7",
      "Viewing Mode": 0
    },
    {
      "Image Name": "image8",
      "Viewing Mode": 0
    },
    {
      "Image Name": "image9",
      "Viewing Mode": 0
    }
  ]
}
```

## Experiment
* This program assumes an experiemental setup with two identical monitors.
* Left and right arrow keys are used to answer, or gamepad's X and Y buttons.

`This program was created by Katya Kozlovsky under The Centre for Vision Research at York University, Toronto, Canada.`
