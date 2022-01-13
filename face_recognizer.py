import cv2

# Local Binary Patterns Histograms

recognizer = cv2.face.LBPHFaceRecognizer_create() 

 #load trained model
 
recognizer.read("E:\\Python Project\\AI Assistant\\face Recognition\\trainer\\trainer.yml") 
 
cascadePath = "E:\\Python Project\\AI Assistant\\face Recognition\\haarcascade_frontalface_default.xml"

 #initializing haar cascade for object detection approach
 
faceCascade = cv2.CascadeClassifier(cascadePath)

#denotes the font type

font = cv2.FONT_HERSHEY_SIMPLEX 

 #number of persons you want to Recognize
 
id = 1

#names, leave first empty bcz counter starts from 0

names = ['','Suvro','Debdas']  

#cv2.CAP_DSHOW to remove warning

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

# set video FrameWidht

cam.set(3, 640)

# set video FrameHeight

cam.set(4, 480) 

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)

minH = 0.1*cam.get(4)

# flag = True

while True:
    
    #read the frames using the above created object
    
    ret, img =cam.read() 
    
    #The function converts an input image from one color space to another
    
    converted_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  

    faces = faceCascade.detectMultiScale( 
        converted_image,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) #used to draw a rectangle on any image
        
        #to predict on every single image
        
        id, accuracy = recognizer.predict(converted_image[y:y+h,x:x+w]) 

        # Check if accuracy is less them 100 ==> "0" is perfect match 
        
        if (accuracy < 100):
            
            id = names[id]
            
            accuracy = "  {0}%".format(round(100 - accuracy))

        else:
            id = "unknown"
            
            accuracy = "  {0}%".format(round(100 - accuracy))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        
        cv2.putText(img, str(accuracy), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 
    
    # Press 'ESC' for exiting video
    
    k = cv2.waitKey(10) & 0xff 
    
    if k == 27:
        
        break
    
print("Thanks for using this program, have a good day.")

cam.release()
cv2.destroyAllWindows()