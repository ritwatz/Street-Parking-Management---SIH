import cv2
import numpy as np
import cvzone
import pickle


cap = cv2.VideoCapture('easy1.mp4')
drawing=False
area_names=[]
polylines=[]
points=[]
current_name=" "


def draw(event,x,y,flags,param):
    global points,drawing
    drawing=True
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x,y)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame=cv2.resize(frame,(1020,500))
    
    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME',draw)
    if cv2.waitKey(1) & 0xFF==ord('d'):
        break
cap.release()
cv2.destroyAllWindows()