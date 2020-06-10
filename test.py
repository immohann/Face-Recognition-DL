import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json 
import random 
import cv2
import numpy as np
from keras.preprocessing import image
import os



#load saved model
model = load_model('mymodel2.h5')

# For naming the classes
l_ = []
for f in os.listdir('dataset/train/'):
    l_.append(f.upper())

l_ = sorted(l_)
people = {}
for i,person in enumerate(l_):
    people[i] = person.title()


#testing
# Loading the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# def face_extractor(img):
#     # Function detects faces and returns the cropped face
#     # If no face detected, it returns the input image
    
#     faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
#     if faces is ():
#         return None
    
#     # Crop all faces found
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
#         cropped_face = img[y:y+h, x:x+w]

#     return cropped_face


# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    cropped_face=None
#     face=face_extractor(frame)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)  
    if faces is ():
        face= None  
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,200,0),2)
        cropped_face = frame[y:y+h, x:x+w]
    face=cropped_face
        
        
    cv2.putText(frame,"Press 'q' to quit", (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 2)
    
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)

        pid = np.argmax(pred,axis=1)[0]
        name="None matching"
        name = people[pid]
            
        cv2.putText(frame,name, (x, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
    else:
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,220),2)
            cropped_face = frame[y:y+h, x:x+w]
        face=cropped_face
        name='Not-Recognized'
        cv2.putText(frame,name, (x, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,220), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

