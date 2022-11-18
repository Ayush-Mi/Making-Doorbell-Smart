# Making Doorbell Smart

## Description

This repo implements a smart intruder detection system using Deep Learning based methods. It can be used as a security camera in front of the door to monitor unknown visitors at home. The whole code can be easily deployed to any edge compute device like raspberry pi having more than 2gb of RAM.

## Method

Here the image from the camera feed is first passed through a face detection model to detect all the faces in the given frame. These detected faces are then passed through FaceNet model which takes in an image with (160,160,3) dimensions and generates a (512,) embeddings. 

#### Architecture
![](https://github.com/Ayush-Mi/Making-Doorbell-Smart/blob/main/Architecture.jpeg)

## Dataset

## Requirements

## Result

#### Given Image

#### Detected Faces

#### Generated Metadata

## Limitations
Being light dependent, the efficiency during night or people standing far away can affect its performance. Also, the model and methods used here are old and newer model can more accurately detect faces in different conditions.

## Future Works
The code can be modified to give SMS/Email based alerts when it detects an unknown person in front of the camera. It can also be intgrated with cloud platform or as simple as Google Sheets to store the data collected in realtime.
