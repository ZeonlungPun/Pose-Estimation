import cv2,cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap=cv2.VideoCapture("E:\opencv\\blink.mp4")
detector=FaceMeshDetector(maxFaces=1)
plotY=LivePlot(640,360,[30,50],invert=True)

idList=[22,23,24,26,110,157,158,159,160,161,130,243]

blinkCounter=0
ratioList=[]
counter=0
color=(255,0,255)
while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) ==cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)


    success,img=cap.read()
    img,faces=detector.findFaceMesh(img,draw=False)

    if faces:
        #找到第一张脸
        face=faces[0]
        for id in idList:
            cv2.circle(img,(face[id][0],face[id][1]),5,(255,0,255),cv2.FILLED)
        #找出特殊标记点
        leftUp=tuple(face[159])
        leftDown=tuple(face[23])
        leftLeft=tuple(face[130])
        leftRight=tuple(face[243])
        lengthVer,_=detector.findDistance(leftUp,leftDown)
        lengthHor,_=detector.findDistance(leftLeft,leftRight)
        ratio=(lengthVer/lengthHor)*100
        # smooth the value
        ratioList.append(ratio)
        if len(ratioList)>10:
            ratioList.pop(0)
        ratioAVG=sum(ratioList)/len(ratioList)
        #there are multiple frames that the value goes down and comes back
        # find the first one and wait 10 frames
        if ratioAVG<39.5 and counter==0:
            color=(0,200,0)
            blinkCounter+=1
            counter=1
        if counter !=0:
            counter+=1
            if counter>10:
                counter=0
                color=(255,0,255)
        cvzone.putTextRect(img,f'blink count:{blinkCounter}',(50,100),colorR=color)

        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)
        imgPlot=plotY.update(ratioAVG,color)
        img=cv2.resize(img,(640,360))
        imgStack=cvzone.stackImages([img,imgPlot],2,1)
    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)






    cv2.imshow('img',imgStack)
    key=cv2.waitKey(40)
    if key==27:
        break
