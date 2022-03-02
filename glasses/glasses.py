import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def detector(img,classifier, scaleFactor=None,minNeighbors=None):
    result=img.copy()
    img2=cv2.imread("dealwithit.png")
    rects=[]
    rects.append(classifier.detectMultiScale(result,scaleFactor=scaleFactor,minNeighbors=minNeighbors))
    coords=[]
    rects.append(classifier.detectMultiScale(result,scaleFactor=scaleFactor,minNeighbors=minNeighbors))             
    for rect in rects:
        for(x,y,w,h) in rect:
            cv2.rectangle(result, (x,y),(x+w,y+h),(255,255,255))
            coords.append([int(x+(w/2)),int(y+(h/2))])
    eyes=[0,1]
    if(len(coords)>1):
        min_distance=result.shape[0]*result.shape[1]
        for i in range(len(coords)-1):
            for j in range(i+1,len(coords)):
                if(coords[i]!=coords[j] and math.sqrt((coords[i][0]-coords[j][0])*(coords[i][0]-coords[j][0])+(coords[i][1]-coords[j][1])*(coords[i][1]-coords[j][1]))<min_distance):
                    min_distance=math.sqrt((coords[i][0]-coords[j][0])*(coords[i][0]-coords[j][0])+(coords[i][1]-coords[j][1])*(coords[i][1]-coords[j][1]))
                    eyes=[i,j]
        eyes[0]=coords[eyes[0]]

        eyes[1]=coords[eyes[1]]
        baser=eyes[0]
        for coo in eyes:
            if(coo[0]<baser[0]):
                baser=coo
        #print(min_distance)
        if(min_distance<1000):
            resizer=(min_distance*0.0015)
            
            #print('resizer:',resizer)
            img2=cv2.resize(img2, (0,0), fx=resizer, fy=resizer)
            baser[0]=int(baser[0]-img2.shape[1]/3)
            baser[1]=int(baser[1]-img2.shape[0]/2)
            #img2[img2==255]=[-1,-1,-1]
            for i in range(img2.shape[0]):
                for j in range(img2.shape[1]):
                    if(img2[i][j][0]!=255):
                        if(i+baser[1]<result.shape[0] and j+baser[0]<result.shape[1] and i+baser[1]>=0 and j+baser[0]>=0):
                            result[i+baser[1]][j+baser[0]]=img2[i][j]
        else:
            pass

    return result

conf=cv2.imread("solvay_conference.jpg")
cooper=cv2.imread("cooper.jpg")
numberator=cv2.imread("car_plate.jpg")
glasses=cv2.imread("dealwithit.png")

face_cascade="haarcascades/haarcascade_frontalface_default.xml"
eye_cascade="haarcascades/haarcascade_eye.xml"
lbp_cascade="lbpcascades/lbpcascade_frontalface.xml"
number_cascade="haarcascades/haarcascade_russian_plate_number.xml"
smile_cascade="haarcascades/haarcascade_smile.xml"
glass_cascade="haarcascades/haarcascade_eye_tree_eyeglasses.xml"

face_classifier=cv2.CascadeClassifier(face_cascade)
result=detector(conf, face_classifier, 1.2, 5)

number_classifier=cv2.CascadeClassifier(number_cascade)
result=detector(numberator, number_classifier, 1.2, 5)

smile_classifier=cv2.CascadeClassifier(smile_cascade)
glasses_classifier=cv2.CascadeClassifier(glass_cascade)
eye_classifier=cv2.CascadeClassifier(eye_cascade)


cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
camera=cv2.VideoCapture(0)
while camera.isOpened():
    ret,frame=camera.read()
    
    
    result=detector(frame, glasses_classifier, 1.2, 5)
    
    
    key=cv2.waitKey(1)
    if(key==ord('q')):
        break
    cv2.imshow('Camera', result)
camera.release()
cv2.destroyAllWindows()