import cv2
#img_car =r"\AI_projects\traffic.jpg"

#loading Cars & pedestrian training data
car_data = r'\xml_files\har_car.xml'
hooman_data = r'\xml_files\hooman.xml'

#video
video = cv2.VideoCapture(r'\images&videos\dashcam.mp4')

#create open cv image
#img= cv2.imread(img_car)
 
#create classifier
car_tracker = cv2.CascadeClassifier(car_data)
pedestrian_tracker = cv2.CascadeClassifier(hooman_data)

#loop
while True:
    #reading frames
    ret, frames = video.read()
    #convert to grayscale
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    #Detect Cars & pedestrians
    cars = car_tracker.detectMultiScale(gray)
    pedestrian = pedestrian_tracker.detectMultiScale(gray)
    #draw rectangles around cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frames, (x,y),(x+w,y+h),(0,255,0),2)
        
    #draw rectangles around pedestrians
    for(x,y,w,h) in pedestrian:
        cv2.rectangle(frames, (x,y),(x+w,y+h),(0,0,128),2)
        cv2.imshow('car_image',frames)

    #display image
    cv2.imshow('car_image',frames)
    key=cv2.waitKey(1)

    #quitting the program by pressing Q
    if key== 81 or key== 113:
        break  


