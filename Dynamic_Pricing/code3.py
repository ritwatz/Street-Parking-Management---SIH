import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import random
import torch

torch.cuda.set_device(0) 

# Check if GPU is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Load the YOLO model
model = YOLO("yolov8s.pt")

# Move the model to the specified device
model.to(device)


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
start_time = time.time()
true_time1 = true_time2 = true_time3 = true_time4 = true_time5 = true_time6 = true_time7 = true_time8 = true_time9 = true_time10 = true_time11 = true_time12 = 0.0
false_time = 0.0
total_price = 0
price_per_hour=0
def price(num_spots, max_hours):
    if num_spots<=12 and num_spots>8:
        price_per_hour=20
    elif num_spots>4 and num_spots<=8:
        price_per_hour=25
    elif num_spots>2 and num_spots<=4:
        price_per_hour=30
    else:
        price_per_hour=35
    total_price = max_hours*price_per_hour/60
    return total_price
cost = {}
for i in range(12):
    cost[i] = 0

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('sample_video.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

area1 = [(506, 326), (596, 319), (622, 363), (502, 367)]

area2 = [(598, 320), (626, 364), (697, 356), (666, 307)]

area3 = [(675, 319), (709, 359), (769, 352), (722, 319)]

area4 = [(749, 315), (785, 354), (834, 347), (798, 295)]

area5 = [(801, 297), (846, 290), (895, 335), (850, 341)]

area6 = [(854, 290), (898, 335), (941, 330), (902, 291)]

area7 = [(424, 366), (442, 346), (302, 339), (276, 363)]

area8 = [(440, 347), (313, 340), (330, 300), (449, 311)]

area9 = [(419, 294), (449, 290), (466, 269), (431, 273)]

area10 = [(370, 293), (393, 275), (423, 275), (409, 294)]

area11 = [(361, 292), (384, 278), (352, 277), (330, 291)]

area12 = [(289, 294), (318, 277), (346, 278), (319, 294)]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    time.sleep(1)
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a.cpu().numpy()).astype("float")

    #    print(px)
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []
    list10 = []
    list11 = []
    list12 = []

    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
            if results1 >= 0:

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list1.append(c)
                cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            results2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
            if results2 >= 0:

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list2.append(c)

            results3 = cv2.pointPolygonTest(np.array(area3, np.int32), ((cx, cy)), False)
            if results3 >= 0:

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list3.append(c)
            results4 = cv2.pointPolygonTest(np.array(area4, np.int32), ((cx, cy)), False)
            if results4 >= 0:

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list4.append(c)
            results5 = cv2.pointPolygonTest(np.array(area5, np.int32), ((cx, cy)), False)
            if results5 >= 0:

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list5.append(c)
            results6 = cv2.pointPolygonTest(np.array(area6, np.int32), ((cx, cy)), False)
            if results6 >= 0:

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list6.append(c)
            results7 = cv2.pointPolygonTest(np.array(area7, np.int32), ((cx, cy)), False)
            if results7 >= 0:

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list7.append(c)
            results8 = cv2.pointPolygonTest(np.array(area8, np.int32), ((cx, cy)), False)
            if results8 >= 0:

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list8.append(c)
            results9 = cv2.pointPolygonTest(np.array(area9, np.int32), ((cx, cy)), False)
            if results9 >= 0:

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list9.append(c)
            results10 = cv2.pointPolygonTest(np.array(area10, np.int32), ((cx, cy)), False)
            if results10 >= 0:

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list10.append(c)
            results11 = cv2.pointPolygonTest(np.array(area11, np.int32), ((cx, cy)), False)
            if results11 >= 0:

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list11.append(c)
            results12 = cv2.pointPolygonTest(np.array(area12, np.int32), ((cx, cy)), False)
            if results12 >= 0:

                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list12.append(c)

    a1 = (len(list1))
    a2 = (len(list2))
    a3 = (len(list3))
    a4 = (len(list4))
    a5 = (len(list5))
    a6 = (len(list6))
    a7 = (len(list7))
    a8 = (len(list8))
    a9 = (len(list9))
    a10 = (len(list10))
    a11 = (len(list11))
    a12 = (len(list12))
    o = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12)
    space = (12 - o)
    print(space)
    if a1 == 1:
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)

        current_time1 = time.time()
        true_time1 += current_time1 - start_time
        start_time = current_time1
        c1 = price(space, true_time1)
        cost[0] = c1
    else:
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)

        cost[0] = 0
    if a2 == 1:
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2)

        current_time2 = time.time()
        true_time2 += current_time2 - start_time
        start_time = current_time2
        c2 = price(space, true_time2)
        cost[1] = c2
    else:
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)

        cost[1]=0
    if a3 == 1:
        cv2.polylines(frame, [np.array(area3, np.int32)], True, (0, 0, 255), 2)

        current_time3 = time.time()
        true_time3 += current_time3 - start_time
        start_time = current_time3
        c3 = price(space, true_time3)
        cost[2] = c3
    else:
        cv2.polylines(frame, [np.array(area3, np.int32)], True, (0, 255, 0), 2)

        cost[2]=0
    if a4 == 1:
        cv2.polylines(frame, [np.array(area4, np.int32)], True, (0, 0, 255), 2)

        current_time4 = time.time()
        true_time4 += current_time4 - start_time
        start_time = current_time4
        c4 = price(space, true_time4)
        cost[3] = c4
    else:
        cv2.polylines(frame, [np.array(area4, np.int32)], True, (0, 255, 0), 2)

        cost[3]=0
    if a5 == 1:
        cv2.polylines(frame, [np.array(area5, np.int32)], True, (0, 0, 255), 2)

        current_time5 = time.time()
        true_time5 += current_time5 - start_time
        start_time = current_time5
        c5 = price(space, true_time5)
        cost[4] = c5
    else:
        cv2.polylines(frame, [np.array(area5, np.int32)], True, (0, 255, 0), 2)

        cost[4]=0
    if a6 == 1:
        cv2.polylines(frame, [np.array(area6, np.int32)], True, (0, 0, 255), 2)

        current_time6 = time.time()
        true_time6 += current_time6 - start_time
        start_time = current_time6
        c6 = price(space, true_time6)
        cost[5] = c6
    else:
        cv2.polylines(frame, [np.array(area6, np.int32)], True, (0, 255, 0), 2)

        cost[5]=0
    if a7 == 1:
        cv2.polylines(frame, [np.array(area7, np.int32)], True, (0, 0, 255), 2)

        current_time7 = time.time()
        true_time7 += current_time7 - start_time
        start_time = current_time7
        c7 = price(space, true_time7)
        cost[6] = c7
    else:
        cv2.polylines(frame, [np.array(area7, np.int32)], True, (0, 255, 0), 2)

        cost[6]=0
    if a8 == 1:
        cv2.polylines(frame, [np.array(area8, np.int32)], True, (0, 0, 255), 2)

        current_time8 = time.time()
        true_time8 += current_time8 - start_time
        start_time = current_time8
        c8 = price(space, true_time8)
        cost[7] = c8
    else:
        cv2.polylines(frame, [np.array(area8, np.int32)], True, (0, 255, 0), 2)

        cost[7]=0
    if a9 == 1:
        cv2.polylines(frame, [np.array(area9, np.int32)], True, (0, 0, 255), 2)

        current_time9 = time.time()
        true_time9 += current_time9 - start_time
        start_time = current_time9
        c9 = price(space, true_time9)
        cost[8] = c9
    else:
        cv2.polylines(frame, [np.array(area9, np.int32)], True, (0, 255, 0), 2)

        cost[8]=0
    if a10 == 1:
        cv2.polylines(frame, [np.array(area10, np.int32)], True, (0, 0, 255), 2)

        current_time10 = time.time()
        true_time10 += current_time10 - start_time
        start_time = current_time10
        c10 = price(space, true_time10)
        cost[9] = c10
    else:
        cv2.polylines(frame, [np.array(area10, np.int32)], True, (0, 255, 0), 2)

        cost[9]=0
    if a11 == 1:
        cv2.polylines(frame, [np.array(area11, np.int32)], True, (0, 0, 255), 2)

        current_time11 = time.time()
        true_time11 += current_time11 - start_time
        start_time = current_time11
        c11 = price(space, true_time11)
        cost[10] = c11
    else:
        cv2.polylines(frame, [np.array(area11, np.int32)], True, (0, 255, 0), 2)

        cost[10]=0
    if a12 == 1:
        cv2.polylines(frame, [np.array(area12, np.int32)], True, (0, 0, 255), 2)

        current_time12 = time.time()
        true_time12 += current_time12 - start_time
        start_time = current_time12
        c12 = price(space, true_time12)
        cost[11] = c12
    else:
        cv2.polylines(frame, [np.array(area12, np.int32)], True, (0, 255, 0), 2)

        cost[11]=0

    cv2.putText(frame, str(space), (23, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
for x in range(12):
    if cost[x]>0:
        print(f"price of slot {x+1} is {cost[x]}")
cap.release()
cv2.destroyAllWindows()
# stream.stop()
