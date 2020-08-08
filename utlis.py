import cv2
import numpy as np

def getContours(img,cThr=[100,100],showCanny=False,minArea=1000,filter=0,draw=False):
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur= cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny= cv2.Canny(imgBlur,cThr[0],cThr[1])
    kernel= np.ones((5,5))
    imgDial= cv2.dilate(imgCanny,kernel,iterations=3)
    imgThre= cv2.erode(imgDial,kernel,iterations=2)
    if showCanny: cv2.imshow("Canny",imgThre)

    contours,hierarchy= cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalContours= []
    for i in contours:
        area= cv2.contourArea(i)
        if area> minArea:
            peri= cv2.arcLength(i,True)
            approx= cv2.approxPolyDP(i,0.02*peri,True)
            bbox= cv2.boundingRect(approx)
            if filter>0:
                if len(approx) == filter:
                    finalContours.append([len(approx),area,approx,bbox,i])
            else:
                finalContours.append([len(approx),area,approx,bbox,i])

    finalContours= sorted(finalContours,key=lambda x:x[1],reverse=True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)
    return img, finalContours

def reorder(mypoints):
    print(mypoints.shape)
    myNewPoints= np.zeros_like(mypoints)
    mypoints= mypoints.reshape((4,2))
    add= mypoints.sum(1)
    myNewPoints[0]= mypoints[np.argmin(add)]
    myNewPoints[3] = mypoints[np.argmax(add)]
    diff= np.diff(mypoints,axis=1)
    myNewPoints[1] = mypoints[np.argmin(diff)]
    myNewPoints[2] = mypoints[np.argmax(diff)]
    return myNewPoints

def warpImg(img,points,w,h,pad=20):
    #print(points)
    points= reorder(points)

    pts1= np.float32(points)
    pts2= np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix= cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp= cv2.warpPerspective(img,matrix,(w,h))
    imgWarp= imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]
    return imgWarp

def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2+(pts2[1]-pts1[1])**2)**0.5