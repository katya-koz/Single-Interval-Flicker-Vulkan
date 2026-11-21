# Single Interval Flicker Experiment

This project is the implementation of an ISO Flicker Paradigm experiment using Vulkan.

## Installation & Setup Instructions

1. Install [VulkanSDK](https://vulkan.lunarg.com/sdk/home)
    - _Ensure that the 'Shader Toolchain Debug Symbols - 64-bit' option is checked under the 'Select Components' step of the installation._
3. Install [OpenCV](https://opencv.org/releases/)
4. Install [GLFW](https://www.glfw.org/download.html)
5. Update environment variables as follows:

| Variable    | Value                                                   |
| ----------- | ------------------------------------------------------- |
| OpenCV_Path | [Your path to openCV here: eg C:\opencv\build]          |
| GLFW_Path   | [Your path to GLFW here: eg C:\glfw\glfw-3.4.bin.WIN64] |
| Vulkan_Path | [Your path to Vulkan here: eg C:\VulkanSDK\1.4.341.1]   |

## Running + Config

- Ensure that both monitors (are HDR compatible, and) are set to HDR mode in Windows (there will be an error otherwise)
- To run: ` ./SingleIntervalFlicker.exe <config.json> <participants.csv> <trials.csv>`
- Example output (output folder as described in config.json):

```
Participant ID: p001
Participant Age: 19
Participant Gender: F
Group Number: 1
Session Number: 1
Block Number: 1
Display Mode: SDR
Interval Mode: single-interval
Start Time: 2026-08-08 16:03:24

Index,Codec,Image,Answer,Position-X,Position-Y,Mode,Response,Fixation Time,Display Time
0,fraunhofer_b,image0,1,3,1,Mono Left,1,2000,3000
1,dsc,image0,0,3,1,Mono Left,1,2000,3000
```


- Program will prompt participant selection dropdown (populated via participants.csv), block # and session #
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

### Prepare the image directory and record the folder paths in the config file. Each image should have 2 permutations per folder, ie `image0_L.ppm` and `image0_R.ppm`. The degraded images should be organized by their respective codecs, in folders.

```
.
├── reference/
│   ├── image0_L.ppm
│   ├── image0_R.ppm
│   ├── image1_L.ppm
│   └── image1_R.ppm
└── codecs/
    ├── jpeg/
    │   ├── image0_L.ppm
    │   ├── image0_R.ppm
    │   ├── image1_L.ppm
    │   └── image1_R.ppm
    └── dsc/
        ├── image0_L.ppm
        ├── image0_R.ppm
        ├── image1_L.ppm
        └── image1_R.ppm

```

### Prepare a configuration file (i.e. config.json):

- Display Mode:
  - 0 --> SDR
  - 1 --> HDR (preferred, SDR fallback)
- Interval Mode:
  - 0 --> Images are shown sequentially (i.e., [image 1] … [image 2] … [collect response]). The participant **cannot** exit the trial early.
  - 1 --> Single interval mode (images are shown in one section, side by side). Trial **can** be exited early with an early response.
  - **Images are not resized in either mode. Ensure images are properly cropped and sized for single interval mode**

```
{
  "Reference Image Directory": "C:\\testSet\\reference",
  "Condition Image Directory": "C:\\testSet\\codecs",
  "Output Directory": "C:\\testOutput",
  "Target FPS": 30,
  "Flicker Rate (Hz)": 5,
  "Wait Time (s)": 2,
  "Image Time (s)": 3,
  "Display Mode": 0,
  "Interval Mode": 1

}
```

### Prepare a trials.csv file, creating the order of trials, and indicate which image should flicker with the 'answer' column. Codec must refer to an existing folder with the same name. Image name must exist in corresponding codec folder. Indicate x and y position of fixation (if applicable, or just 0,0)

- Stereo mode:
  - 0 --> Stereo
  - 1 --> Mono Left
  - 2 --> Mono Right

```
codec,image_name,answer,x,y,stereo mode
jpeg,image0,1,3,1,1
dsc,image0,1,3,1,1
jpeg,image1,1,3,1,0
dsc,image1,1,3,1,0
jpeg,image2,1,3,1,0
dsc,image2,1,3,1,0
...
```

### Prepare particiapants.csv file. Rows in the participant csv will automatically populate the dropdown in the dialog window popup at start of block.

```
subject_id,age,gender,group
p001,19,F,1
p002,23,M,1
p003,18,F,2
```

## Experiment

- This program assumes an experimental setup with two identical monitors.
- Left and right arrow keys are used to answer, or gamepad's X and Y buttons.

`This program was created by Katya Kozlovsky under The Centre for Vision Research at York University, Toronto, Canada.`
