import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import time
import cv2
import matplotlib.pyplot as plt
import os
import datetime
import uuid
from face_detect import face_detect
from face_rec import face_rec

# Function to save the detected face image data and metadata 
def save_data(img,file_name,timestamp,date,repeat):
    if os.path.exists('./master_file.csv'):
        df = pd.read_csv('./master_file.csv',index_col=[0])
    else:
        df = pd.DataFrame(columns=['file_path','time','date','repeat'])
    df2 = {'file_path':file_name,'time':timestamp,'date':date,'repeat':repeat}
    df = df.append(df2,ignore_index=True)
    df.to_csv('master_file.csv')
    
    if not repeat:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(file_name,img)
    print("file saved at {}".format(file_name))

#capturing video camera footage
video = cv2.VideoCapture(0)
# Loading Face detection model
model = face_detect(model='res10_300x300_ssd_iter_140000.caffemodel',proto='deploy.prototxt.txt')
# Loading Face recognition model
detect = face_rec('./image_db/')
repeat_detect = face_rec('./unknown_faces/')

init_time = time.time()

while True:
    ret, frame = video.read()
    if ret:
        img = cv2.imwrite('tmp.jpg',frame)
        x,y = model.detect_face('tmp.jpg')
        for i in x.keys():
            img = cv2.resize(x[i],(160,160))
            img = np.expand_dims(img,axis=0)
            
            if not detect.face_check(img):
                if not repeat_detect.face_check(img):
                    print("Unknown face detected")
                    data = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f_name = './unknown_faces/'+str(uuid.uuid4())+'.jpg'
                    save_data(img[0],file_name=f_name,timestamp=str(data[11:]),date= str(data[:10]),repeat=False)
                elif (time.time() - init_time) >=300:
                    init_time = time.time()
                    print('repeat_detect')
                    data = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f_name = './unknown_faces/'+str(uuid.uuid4())+'.jpg'
                    save_data(img[0],file_name=f_name,timestamp=str(data[11:]),date= str(data[:10]),repeat=True)

        os.remove('tmp.jpg')

