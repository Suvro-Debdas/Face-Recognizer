import cv2

#create a video capture object which is helpful to capture videos through webcam.

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# set video FrameWidth

cam.set(3, 640) 

# set video FrameHeight

cam.set(4, 480)

detector = cv2.CascadeClassifier("E:\\Python Project\\AI Assistant\\face Recognition\\haarcascade_frontalface_default.xml")

#Haar Cascade classifier is an effective object detection approach

face_id = input("Enter a Numeric user ID  here:")

#Use integer ID for every new face (0,1,2,3,4,5,6,7,8,9........)

print("Taking samples, look at camera ....... ")

count = 0 # Initializing sampling face count

while True:
    
    #read the frames using the above created object
    
    ret, img = cam.read() 
    
    #The function converts an input image from one color space to another
    
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = detector.detectMultiScale(converted_image, 1.3, 5)

    for (x,y,w,h) in faces:
        
        #used to draw a rectangle on any image
        
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) 
        
        count += 1
        
        # To capture & Save images into the datasets folder
        
        cv2.imwrite("E:\\Python Project\\AI Assistant\\face Recognition\\samples\\face." + str(face_id) + "." + str(count) + ".jpg", converted_image[y:y+h,x:x+w])
        
        #Used to display an image in a window

        cv2.imshow("E:\\Python Project\\AI Assistant\\face Recognition\\image", img) 
        
    # Waits for a pressed key
    
    k = cv2.waitKey(100) & 0xff 
    
    # Press 'ESC' to stop
    
    if k == 27: 
        
        break
    
    # Take 50 sample (More sample --> More accuracy)
    
    elif count >= 10: 
        
         break

print("Samples taken now closing the program....")
cam.release()
cv2.destroyAllWindows()
