import cv2
import time
import pandas as pd
import numpy as np

"""
The below python object is to detect faces within a given image.
It returns a dictionary of detected faces in the image and the original
image with bounding boxes around detected ones.
"""
class face_detect:
    def __init__(self, model,proto):
        self.detector = cv2.dnn.readNetFromCaffe(proto,model)

    def detect_face(self,image):
        og_image = cv2.imread(image)
        og_size = og_image.shape
        cp_image = og_image.copy()
        cp_image = cv2.resize(cp_image,(300,300))

        aspect_ratio_x = (og_size[1] / 300)
        aspect_ratio_y = (og_size[0] / 300)

        imageBlob = cv2.dnn.blobFromImage(image = cp_image)

        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        detections = detections[0][0]
        detections = [detections[x] for x in range(detections.shape[0]) if detections[x][2]>0.9]

        no_detections = {}
        count = 0
        for i in detections:
            count += 1
            img_id = i[0]
            is_face = i[1] #face or background
            confidence_score = str(round(100*i[2],2))+"%"

            left = i[3]* 300
            top = i[4] * 300
            right = i[5] * 300
            bottom = i[6] * 300

            detected_face = og_image[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
            no_detections[count] = detected_face[:,:,::-1]
            if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
                cv2.putText(og_image, confidence_score, (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(og_image, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (255, 255, 255), 1) #draw rectangle to main image
        
        return no_detections, og_image[:,:,::-1]