# handtracking.py
import cv2
import mediapipe as mp

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
    
    def fingersUp(self, lmList):
        fingers = []
        if len(lmList) != 0:
            for id in range(8, 21, 4):
                if lmList[id][2] < lmList[id][2]+0.5*(lmList[id - 2][2]-lmList[id][2]):  # index start from top
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def recognizeGesture(self, lmList):
        fingers = self.fingersUp(lmList)
        if abs(lmList[5][1]-lmList[3][1]) <100 and abs(lmList[6][1]-lmList[4][1]) <80 and fingers==[1,1,1,1]:
            return "Hii Ravi"
        elif fingers==[0,1,1,1] and lmList[4][2]-lmList[8][2] and abs(lmList[4][1]-lmList[8][1])<20:
                return 'Superb'    
        elif fingers==[1,0,0,0] and abs(lmList[4][2]-lmList[8][2]) >abs(lmList[10][1]-lmList[5][1]) and lmList[4][1]>lmList[10][1] :
            return "Warning"
        elif fingers == [0,0,0,0] and lmList[4][2]<lmList[3][2]:
            return "Thumbs Up!"
        elif fingers == [0,0,0,0] and lmList[4][2]>lmList[3][2] and lmList[20][2]<lmList[1][2]:
            return "Thumbs Down!" 
        elif fingers == [1,1,0,0] and lmList[8][1]-lmList[12][1]>lmList[6][1]-lmList[10][1]:
            return "Peace !"       
        elif fingers == [1,1,0,0] and lmList[12][1]>lmList[8][1]:
            return "Promise" 

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector(detectionCon=0.6, maxHands=1)

    while True:
        success, img = cap.read()
        img = detector.findHands(img,draw=True)
        lmList = detector.findPosition(img,draw=True)

        if len(lmList) != 0:
            gesture = detector.recognizeGesture(lmList)
            cv2.putText(img, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
