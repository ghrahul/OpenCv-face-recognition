# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:08:56 2018

@author: rahul
"""

import cv2
import numpy as np
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)  
id = input("Enter user id ")
samplenum = 0
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        samplenum = samplenum+1
        cv2.imwrite("dataset/user." + str(id)+"." + str(samplenum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)
        
    cv2.imshow("Face",img)
    if(samplenum>100):
        break
    
cam.release()
cv2.destroyAllWindows()