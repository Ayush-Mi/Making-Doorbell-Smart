from keras_facenet import FaceNet
from scipy import spatial
import glob
import cv2
import numpy as np

"""
This is object for comparing given two face images.
It uses Keras-Facenet to generate embeddings for the 
given face image and caluculates cosine distance between them.
The threshold is set to 0.75 but can be adjusted as per 
its performance.
"""
class face_rec:
    def __init__(self,face_db):
        self.embedder = FaceNet()
        self.folder = face_db
    
    def similarity(self,vec1,vec2):
        return 1 - spatial.distance.cosine(vec1,vec2)
    
    def pre_process(self,img):
        img = cv2.imread(img)
        img.resize(160,160,3)
        img = np.expand_dims(img,axis=0)
        return img

    def face_check(self,test_image):
        test_image = test_image #self.pre_process(test_image)
        for image in glob.glob(self.folder+'/*.jpg'):
            tmp_img = self.pre_process(image)

            dist = self.similarity(self.embedder.embeddings(tmp_img)[0],
                                    self.embedder.embeddings(test_image)[0])
            
            if dist>0.75:
                return True
            
        return False

