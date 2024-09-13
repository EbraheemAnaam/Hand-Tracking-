import cv2
import mediapipe as mp
import time

class handDetector():
    
    def __init__(self, mode=False, maxHands=2,  modelComplexity=1, detectionCon=0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, drow=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
            
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                 if drow:
                     self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) 
        return img 

    def findPosition(self, img, handNo = 0, drow=  True):
        
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                #if id == 8 :
                if drow:
                    cv2.circle(img, (cx, cy), 5, (255,8,255), cv2.FILLED)
        return lmList    

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True :
        success, img = cap.read()
        img = cv2.flip(img,1)
        img = detector.findHands(img, drow=False)
        lmList= detector.findPosition(img, drow=False) 
        if len(lmList) != 0 :
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
    
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,8,255), 10)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q") :
            break
        
        
if __name__=="__main__":
    main()