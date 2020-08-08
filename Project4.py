import cv2
import numpy as np
import requests
import utlis
#################################

webcam= False
path= "Resources/Image.jpeg"
cam= cv2.VideoCapture(0)
cam.set(3,1920)
cam.set(4,1080)
cam.set(10,150)
scale= 5
wP= 210*scale
hP= 297*scale


while True:
    #if webcam:
    img_res = requests.get("http://192.168.43.1:8080/shot.jpg")
    img_arr = np.array(bytearray(img_res.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    #else: img= cv2.imread(path)

    img, conts= utlis.getContours(img,minArea=50000,filter=4)
    if len(conts) !=0:
        biggest= conts[0][2]
        #print(biggest)
        imgWarp= utlis.warpImg(img,biggest,wP,hP)
        img2, conts2 = utlis.getContours(imgWarp,cThr=[50,50],minArea=2000,filter=4,draw=False)
        if len(conts) !=0:
            for obj in conts2:
                cv2.polylines(img2,[obj[2]],True,(0,255,0),2)
                nPoints= utlis.reorder(obj[2])
                nW= round((utlis.findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
                nH = round((utlis.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)
                cv2.arrowedLine(img2,(nPoints[0][0][0],nPoints[0][0][1]),(nPoints[1][0][0],nPoints[1][0][1]),
                                (255,0,255),3,6,0,0.1)
                cv2.arrowedLine(img2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 6, 0, 0.1)
                x, y, w, h = obj[3]
                cv2.putText(img2,"{}cm".format(nW),(x+30,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1,(255,0,255),2)
                cv2.putText(img2,"{}cm".format(nH),(x-70,y+h//2),cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1,(255, 0, 255),2)
        cv2.imshow("A4",img2)
    #img = cv2.resize(img, (0,0),None,0.5,0.5)
    cv2.imshow("Output",img)

    cv2.waitKey(1)