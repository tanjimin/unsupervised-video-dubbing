# Post-Processing Stage

##Requirement

- LibROSA 0.7.2 
- dlib 19.19
- OpenCV 4.2.0

- Pillow 6.2.2
- PyTorch 1.2.0
- TorchVision 0.4.0



##Folder Structure

```
.
├── source                  
│   ├── audio_driver_mp4    # contain audio drivers (saved in mp4 format)
│   ├── audio_driver_wav    # contain audio drivers (saved in wav format)
│   ├── base_video          # contain base videos (videos you'd like to modify)
│   ├── dlib            		# trained dlib models
│   └── model               # trained landmark generation models
├── main.py									# main function for post processing
├── main_support.py					# support functions used in main.py
├── models.py								# define the landmark generation model
├── compare_openness.ipynb	# mouth openness comparison across generated videos
└── README.md
```

> - shape_predictor_68_face_landmarks.dat
>
> This is trained on the ibug 300-W dataset (https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
>
> The license for this dataset excludes commercial use and Stefanos Zafeiriou, one of the creators of the dataset, asked me to include a note here saying that the trained model therefore can't be used in a commerical product. So you should contact a lawyer or talk to Imperial College London to find out if it's OK for you to use this model in a commercial product.
>
> {C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 300 faces In-the-wild challenge: Database and results. Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.}



##Detailed steps for post processing

- Usage: ```python3 main.py -r step  ```
  - e.g: `python3 main.py -r 1` will run the first step and etc 

####Step 1 — generate landmarks

- Input
  - Base video file path (`./source/base_video/base_video.mp4`)
  - Audio driver file path (`./source/audio_driver_wav/audio_driver.wav`)
  - Epoch (`int`)
- Output (`./result`)
  - keypoints.npy (# generated landmarks in `npy` format)
  - source.txt (contains information about base video, audio driver, model epoch)
- Process
  - Extract facial landmarks from base video
  - Extract MFCC features from driver audio
  - Pass MFCC features and facial landmarks into the model to retrieve mouth landmarks
  - Combine facial & mouth landmarks and save in `npy` format

#### Step 2 — Test generated frames

- Input
  - None
- Output (`./result`)
  - Folder — save_keypoints: visualized generated frames
  - Folder — save_keypoints_csv : landmark coordinates for each frame, saved in `txt` format
  - openness.png: mouth openness measured and plotted across all frames
- Process
  - Generate images from `npy` file
  - Generate openness plot

#### Step 3 — Generate modified videos with sound

- Input
  - Saved frames folder path
    - By default, it is saved in `./result/save_keypoints`; you can enter `d` to go with default path
    - Otherwise, input the frames folder path
  - Audio driver file path (`./source/audio_driver_wav/audio_driver.wav`)
- Output (`./result/save_keypoints/result/`)
  - video_without_sound.mp4: modified videos without sound
  - audio_only.mp4: audio driver
  - final_output.mp4: modified videos with sound
- Process
  - Generate the modified video without sound with define fps
  - Extract `wav` from audio driver
  - Combine audio and video to generate final output

## Important Notice

- You may need to modify how MFCC features are extracted in `extract_mfcc` function
  - Be careful about sample rate, window_length, hop_length
  - Good resource: https://www.mathworks.com/help/audio/ref/mfcc.html
- You may need to modify the region of interest (mouth area) in `frame_crop` function
- You may need to modify the frame rate defined in step_3 of the main.py, which should be your base video fps

```python
# How to check your base video fps
# source: https://www.learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/

import cv2
video = cv2.VideoCapture("video.mp4");

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
video.release()
```

- You may need to modify the shell path

```shell
echo $SHELL
```

- You may need to modify the audio sampling rate in `extract_audio` function
- You may need to customize your parameters in `combine_audio_video` function
  - Good resource: https://ffmpeg.org/ffmpeg.html
  - https://gist.github.com/tayvano/6e2d456a9897f55025e25035478a3a50



## Update History

- March 22, 2020: Drafted documentation

