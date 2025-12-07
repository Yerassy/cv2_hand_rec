import cv2
import mediapipe as mp
import time
import math
import numpy as np


class handDetector():
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
            min_tracking_confidence=self.trackCon,
            model_complexity=0  # Добавлено для совместимости
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False  # Оптимизация
        self.results = self.hands.process(imgRGB)
        imgRGB.flags.writeable = True
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Рисуем скелет руки
                    self.mpDraw.draw_landmarks(
                        img, 
                        handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                        self.mpDraw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        
        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])
                    
                    if draw and id in self.tipIds:
                        cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
                        cv2.putText(img, str(id), (cx - 10, cy - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                if xList and yList:
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    bbox = xmin, ymin, xmax, ymax
                    
                    if draw:
                        cv2.rectangle(img, (xmin - 20, ymin - 20), 
                                     (xmax + 20, ymax + 20), (0, 255, 0), 2)
        
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        
        if len(self.lmList) == 0:
            return [0, 0, 0, 0, 0]
        
        # Thumb
        if len(self.lmList) > self.tipIds[0]:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            fingers.append(0)
        
        # 4 Fingers
        for id in range(1, 5):
            if len(self.lmList) > self.tipIds[id]:
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                fingers.append(0)
        
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        if len(self.lmList) > max(p1, p2):
            x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
            x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
                cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
                cv2.putText(img, f'{int(math.hypot(x2-x1, y2-y1))}', 
                           (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            length = math.hypot(x2 - x1, y2 - y1)
            return length, img, [x1, y1, x2, y2, cx, cy]
        
        return 0, img, [0, 0, 0, 0, 0, 0]