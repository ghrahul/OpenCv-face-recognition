# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 20:31:29 2018

@author: rahul
"""

import os
import cv2
import numpy as np
from PIL import Image
recognizer =  cv2.createLBPHFaceRecognizer()
path = 'dataset'
def ImageId(path):
    imagepath = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for images in imagepath: #iamges are allthe imagepaths
        faceImg = Image.open(images).convert('L') #converting in gray scale image
        facenp = np.array(faceImg,'uint8')  #converting images in numpy array
        ID = int(os.path.split(images)[-1].split('.')[1]) #getting the id of each image
        faces.append(facenp) #storing numpy values to faces list
        IDs.append(ID) #storing id to IDs
        cv2.imshow("training",facenp) #shows which image is being trained
        cv2.waitKey(10)
    return np.array(IDs), faces

IDs,faces = ImageId(path)
recognizer.train(faces,IDs) #for training
recognizer.save('recognizer/trainingData.yml') #for saving training data
cv2.destroyAllWindows()

        
        




