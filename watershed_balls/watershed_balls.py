import cv2
import numpy as np


camera=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cv2.namedWindow('Camera',cv2.WINDOW_KEEPRATIO)

lower = (10,60, 60)
upper = (255, 255, 255)

bounds = [[lower,upper]]

while camera.isOpened():
    _, image = camera.read()
    image = cv2.flip(image, 1)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)

    dist = cv2.distanceTransform(mask,cv2.DIST_L2,5)
    reg,fg=cv2.threshold(dist, 0.7*dist.max(),255,0)
    fg=np.uint8(fg)
    confuse=cv2.subtract(mask,fg)
    ret,markers=cv2.connectedComponents(fg)
    markers+=1
    markers[confuse==255]=0
    wmarkers = cv2.watershed(image, markers.copy())
    contours, heirarchy = cv2.findContours(wmarkers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    res = 0
    areas = []
    for i in range(len(contours)):
        if heirarchy[0][i][3] == -1 and heirarchy[0][i][0] != -1 and cv2.contourArea(contours[i])>3000:
            areas.append(cv2.contourArea(contours[i]))
            res+=1
            cv2.drawContours(image, contours, i, (255, 0, 0), 10)
    print(res)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    cv2.imshow("Camera", image)

camera.release()
cv2.destroyAllWindows()



