# Making Doorbell Smart

## Description

This repo implements a smart intruder detection system using Deep Learning based methods. It can be used as a security camera in front of the door to monitor unknown visitors at home. The whole code can be easily deployed to any edge compute device like raspberry pi having more than 2gb of RAM.

## Method

Here the image from the camera feed is first passed through a face detection model to detect all the faces in the given frame. These detected faces are then passed through FaceNet model which takes in an image with (160,160,3) dimensions and generates a (512,) embeddings. This embeddings are then compared with the embeddings of all the known faces in the database and if no matching image is found then it is classified as unknown face. The timestamp of this classification is noted and stared in a csv file which also contains the path where this unkown face image is stored. If this unkonwn person stays within the frame for 5 mins or less then no new entry is created for it, however, if the person reappears or stays more than 5 mins in the frame (which is unlikely) then a new entry is created in the metadata file.

#### Architecture
![](https://github.com/Ayush-Mi/Making-Doorbell-Smart/blob/main/img/Architecture.jpeg)

## Model

For face detection problem I have used pretrained ResNet-10 based Caffe model with input shape of (None,300,300,3). 
For face verification Keras-Facenet Model is used to generate embeddings of 512 dimensions for the detected faces.


## Requirements
The code requires at least 2gb of memory to run inference in near realtime and should support python3.5+ along with tensorflow 2.5+. All the other libraries can be installed from the requirements.txt file

## Steps
- Install all the required libraries from requirements.txt file
- Place the cropped images of known faces in the 'image_db'
- Make sure the camera is attached and is accesible
- Run `python intruder_monitoring.py`
- Wait for few seconds to load the model
- The unknown faces detected will be saved in the Uknown_faces folder and corresponding metadata will be stored at master_file.csv

## Results

#### Given Image
![](https://github.com/Ayush-Mi/Making-Doorbell-Smart/blob/main/img/test_1.jpg)

#### Detected Faces
![](https://github.com/Ayush-Mi/Making-Doorbell-Smart/blob/main/unknown_faces/03d65536-02a1-4791-9188-966e10ab4676.jpg)

![](https://github.com/Ayush-Mi/Making-Doorbell-Smart/blob/main/unknown_faces/d1cd6db9-a7c0-4ebb-b48e-b35f798c69fb.jpg)

![](https://github.com/Ayush-Mi/Making-Doorbell-Smart/blob/main/unknown_faces/db13a86f-b001-4fa3-91d2-2b3d92157370.jpg)

#### Generated Metadata
![](https://github.com/Ayush-Mi/Making-Doorbell-Smart/blob/main/img/master_sheet.png)

## Limitations
Being light dependent, the efficiency during night or people standing far away can affect its performance. Also, the model and methods used here are old and newer model can more accurately detect faces in different conditions.

The system currently gives near realtime inference but the inference speed will be effected negatively once the known face database increases. Hence, it would be ideal for home use cases only or monitoring restricted place which is less accesible

## Future Works
The code can be modified to give SMS/Email based alerts when it detects an unknown person in front of the camera. It can also be intgrated with cloud platform or as simple as Google Sheets to store the data collected in realtime.
