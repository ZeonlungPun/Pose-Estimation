import cv2,time
import numpy as np
import PoseModule as pm

cap=cv2.VideoCapture(0)
width = 1640
ret = cap.set(3, width)
height = 1480
ret = cap.set(4, height)
detector=pm.poseDetector()

count=0
#up:0 down:1
dir=0
pTime=0
while True:
    success,img=cap.read()
    img=detector.findPose(img)
    lmList=detector.findPosition(img,False)
    if len(lmList)!=0:
        #left arm
        angle=detector.findAngle(img,11,13,15)
        #reflect x: (210,310) to y: (0,100)  find the interp num of angle
        per=np.interp(angle,(210,310),(0,100))
        #bar min 650
        bar=np.interp(angle,(210,310),(650,100))

        #check for the dumbbell curls
        if per==100:
            if dir==0:
                count+=0.5
                #which direction we are moving to
                dir=1
        if per==0:
            if dir==1:
                count+=0.5
                dir=0
        #cv2.putText(img,str(count), (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        color = (255, 0, 255)
        # Draw Bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                    color, 4)

        # Draw Curl Count

        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                    (255, 0, 0), 25)


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)



    cv2.imshow("img",img)
    cv2.waitKey(100)
