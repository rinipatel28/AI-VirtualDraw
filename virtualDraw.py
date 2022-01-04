import cv2
import mediapipe as mp
import time
import numpy as np
import os
import HandTrackingMin as rini

brushThickness = 3
drawColor = (0,0,0)
folderPath = "rini"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
bluepen = overlayList[0]
bluepen_selected = overlayList[1]
eraser = overlayList[2]
redpen = overlayList[3]
redpen_selected = overlayList[4]

cap = cv2.VideoCapture(0)
detector = rini.handDetector(detectionCon = 0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((480,640,3),np.uint8)

while True:
    success, img = cap.read()
    #img[0:100,440:540] = bluepen
    #img[0:100,540:640] = redpen
    img = cv2.flip(img,1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList)!= 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        
        fingures = detector.finguresUp()
        #print(fingures)
        
        if fingures[1] and fingures[2]:
            #cv2.rectangle(img, (x1, y1-15), (x2,y2 + 15), (0,0,0) , cv2.FILLED)
            #print("Choose color")
            if y1 < 50 and 440 < x1 < 540:
                drawColor = (0,0,255)
                img[100:150,590:640] = redpen_selected
                brushThickness = 3
            if y1 < 50 and 540 < x1 < 640:
                drawColor = (255,0,0)
                img[100:150,590:640] = bluepen_selected
                brushThickness = 3
            if y1 < 100 and 340 < x1 < 440:
                drawColor = (0,0,0)
                brushThickness = 40
            cv2.rectangle(img, (x1, y1-15), (x2,y2 + 15), drawColor , cv2.FILLED)   
        if fingures[1] and fingures[2]==False:
            cv2.circle(img, (x1,y1), 3 , (0,0,0), cv2.FILLED )
            #print("Drawing mode is on") 
            if xp==0 and yp==0:
                xp, yp = x1, y1
            cv2.line(img, (xp, yp) ,(x1,y1), drawColor , brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor , brushThickness)
            xp, yp = x1, y1
    img[0:100,340:440] = eraser
    img[0:100,440:540] = redpen
    img[0:100,540:640] = bluepen
    img = cv2.addWeighted(img , 0.5 , imgCanvas , 0.5 ,0)
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
    