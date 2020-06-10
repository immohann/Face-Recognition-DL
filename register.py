

#import required libraries
import cv2
import sys , os
import numpy as np
##############################################
#setting up the dataset files
# It captures images and stores them in datasets  
# folder under the folder name of sub_dir 
haar_file = 'haarcascade_frontalface_default.xml'
# All the faces data will be 
#  present this folder 
train = 'datasets/train'  
validation='datasets/validation'
###############################################
# These are sub data sets of folder,  
# for my faces I've used my name you can  
# change the label here 
sub_dir = input('Enter Name to Register Face: ')
###############################################
# setting path for storing the images in train and test set
dataset='dataset'
train = 'dataset/train'  
validation='dataset/validation'

if not os.path.isdir(dataset): 
    os.mkdir(dataset) 

if not os.path.isdir(train): 
    os.mkdir(train) 

if not os.path.isdir(validation): 
    os.mkdir(validation) 

path1 = os.path.join(train, sub_dir) 
path2=os.path.join(validation, sub_dir) 
if not os.path.isdir(path1): 
    os.mkdir(path1) 
if not os.path.isdir(path2): 
    os.mkdir(path2) 
#################################################
# defining the size of images  
(width, height) = (400, 400)  
#################################################
#'0' is used for my webcam,  
# if you've any other camera 
#  attached use '1' like this 
face_cascade = cv2.CascadeClassifier(haar_file) 
webcam = cv2.VideoCapture(0) 
#################################################
# this function is created in order to avoid overwriting of images
# by keeping the count of the files already in the mentioned directory
def get_max(l):
    number = []
    for word in l:
        temp = ''
        for letter in word:
            if letter != '.':
                temp += letter
            else:
                break
        number.append(int(temp))

    return max(number)
#####################################################
# The program loops until it has 50 images of the face. 
path='dataset/train/'+sub_dir
if os.listdir(path):
    count = get_max(os.listdir(path))
else:
    count = 0

num_img = count + 160

split=0.8
mid=num_img//2
split_train=split*num_img
split_test=(1-split)*num_img

count = 1
while count < num_img:  
    (_, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
    
    for (x, y, w, h) in faces: 
        cv2.rectangle(im, (x-20, y-20), (x + w+20, y + h+20), (0, 0, 0), 2) 
        face = gray[y:y + h, x:x + w] 
        face_resize = cv2.resize(face, (width, height)) 
    
        if count>mid and count <mid+split_test:
            cv2.imwrite('% s/% s.jpg' % (path2, count), face_resize)
        else:
            cv2.imwrite('% s/% s.jpg' % (path1, count), face_resize) 
    
        
        
      #  cv2.imwrite('% s/% s.jpg' % (path, count), face_resize) 
    count += 1
      
    cv2.imshow('OpenCV', im) 
    key = cv2.waitKey(10) 
    if key == 27: 
        break
webcam.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")
#######################################################
