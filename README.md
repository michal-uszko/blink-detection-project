# Blink detection with facial landmarks

## General Info
The goal of this project was to make an application, which counts how many times a person blinks based on video stream from webcam.
Whether eyes are closed or open, it is determined by calculating Eye Aspect Ratio (EAR).
For this task, dlib library was used both for face detection and eyes landmarks detection.

## Installation
Following libraries was used for this project:
* dlib (installation: http://dlib.net/compile.html)
* OpenCV (contrib version) (installation: https://pypi.org/project/opencv-python/)
* imutils (installation: ```pip install imutils```)

## Setup
Before running the script, you will need to download ```shape_predictor_68_face_landmarks.dat.bz2``` from here: http://dlib.net/files/

After downloading and unpacking, place the file into the same directory as ```face_detection.py``` file.

Then, run command below (you will need to connect a webcam to do so):
```
python face_detection.py
```
