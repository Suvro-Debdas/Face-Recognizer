import cv2
import numpy as np
from PIL import Image 
import os

# Path for samples already taken

path = "E:\\Python Project\\AI Assistant\\face Recognition\\samples" 

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Local Binary Patterns Histograms

detector = cv2.CascadeClassifier("E:\\Python Project\\AI Assistant\\face Recognition\\haarcascade_frontalface_default.xml")

#Haar Cascade classifier is an effective object detection approach

# function to fetch the images and labels

def Images_And_Labels(path): 

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    
    faceSamples=[]
    
    ids = []
    
    # to iterate particular image path
    
    for imagePath in imagePaths: 
        
        # convert it to grayscale
        
        gray_img = Image.open(imagePath).convert('L') 
        
        #creating an array
        
        img_arr = np.array(gray_img,'uint8') 

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        faces = detector.detectMultiScale(img_arr)

        for (x,y,w,h) in faces:
            
            faceSamples.append(img_arr[y:y+h,x:x+w])
            
            ids.append(id)

    return faceSamples,ids

print ("Training faces. It will take a few seconds. Wait ...")

faces,ids = Images_And_Labels(path)

recognizer.train(faces, np.array(ids))

# Save the trained model as trainer.yml

recognizer.write("E:\\Python Project\\AI Assistant\\face Recognition\\trainer\\trainer.yml")  

print("Model trained, Now we can recognize your face.")