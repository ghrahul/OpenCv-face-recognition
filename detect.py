# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 20:05:02 2018

@author: rahul
"""

import cv2
import numpy as np
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
recognizer =  cv2.createLBPHFaceRecognizer()
recognizer.load("recognizer/trainingData.yml")  
id = 0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,3,1,0,3)
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id = "Rahul"
        elif(id==2):
            id = "Narendra Modi"
        else:
            id = "unknown"
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
    cv2.imshow("Face",img)
    if(cv2.waitKey(1) == ord('q')):
        break
        
cam.release()
cv2.destroyAllWindows()
