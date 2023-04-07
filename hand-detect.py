import os
import cv2,time
import mediapipe as mp
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5,modelC=1):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelC=modelC

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelC,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #detection
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                #draw the detection point
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            #record specific coordinate of different hand position
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList



wCam,hCam=640,480

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath="E:\opencv\gesture"
mylist=os.listdir(folderPath)
#gesture img list
overlayList=[cv2.imread(f'{folderPath}/{imPath}')for imPath in mylist]

pTime=0

detector=handDetector()
tipsId=[4,8,12,16,20]

while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
#left hands
    #tell the figures are closed or not
    if len(lmList)!=0:
        fingers=[]
        # judge by x coordinate thumb
        if lmList[tipsId[0]][1] < lmList[tipsId[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            #judge by y coordinate 4figures
            if lmList[tipsId[id]][2]<lmList[tipsId[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFigures=fingers.count(1)
        #print(fingers)
        #print(totalFigures)
        figure_img=overlayList[totalFigures]
        figure_img=cv2.resize(figure_img,(150,150))
        img[0:150,0:150]=figure_img

        cv2.rectangle(img,(20,275),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFigures),(45,375),cv2.FONT_HERSHEY_PLAIN,
                    10,(255,0,0),25)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS:{int(fps)}',(400,30),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("image",img)
    key=cv2.waitKey(1)
    if key==27:
        break

