import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
from timeit import default_timer as timer
import random


model = YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
#start_time = time.time()
true_time1 = true_time2 = true_time3 = true_time4 = true_time5 = true_time6 = true_time7 = true_time8 = true_time9 = true_time10 = true_time11 = true_time12 = true_time15 = 0.0
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
    elif num_spots<=2:
        price_per_hour=35
    total_price = max_hours*price_per_hour/60
    return np.round(total_price)
cost = {}
for i in range(15):
    cost[i] = 0

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('easy1.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

area1 = [(506, 326), (596, 319), (622, 363), (502, 367)]

area2 = [(602, 313), (629, 364), (699, 360), (670, 308)]

area3 = [(679, 319), (718, 361), (776, 354), (733, 295)]

area4 = [(743, 303), (785, 351), (838, 345), (792, 299)]

area5 = [(794, 294), (846, 288), (894, 337), (853, 343)]

area6 = [(853, 286), (889, 335), (942, 323), (897, 278)]

area7 = [(424, 366), (442, 346), (302, 339), (276, 363)]

area8 = [(440, 347), (313, 340), (330, 300), (449, 311)]

area10 = [(404, 294), (422, 274), (392, 278), (371, 291) ]

area9 = [(414, 291), (451, 292), (464, 263), (433, 263)]

area11 = [(361, 292), (384, 278), (352, 277), (330, 291)]

area12 = [(289, 294), (318, 277), (346, 278), (319, 294)]

area13 = [(284, 294), (309, 276), (282, 277), (250, 291)]

area14 = [(248, 291), (278, 276), (242, 276), (202, 290)]

area15 = [(150, 290), (182, 289), (215, 275), (192, 263)]

while (cap.isOpened()):
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    time.sleep(1)
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
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
    list13 = []
    list14 = []
    list15 = []

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
            results13 = cv2.pointPolygonTest(np.array(area13, np.int32), ((cx, cy)), False)
            if results13 >= 0:
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list13.append(c)
            results14 = cv2.pointPolygonTest(np.array(area14, np.int32), ((cx, cy)), False)
            if results14 >= 0:
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list14.append(c)
            results15 = cv2.pointPolygonTest(np.array(area15, np.int32), ((cx, cy)), False)
            if results15 >= 0:
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list15.append(c)

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
    a13 = (len(list13))
    a14 = (len(list14))
    a15 = (len(list15))
    o = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15)
    space = (15 - o)

    if a1 == 1:
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)

        current_time1 = time.time()
        true_time1 += current_time1 - start_time
        start_time = current_time1
        cost[0] = price(space, true_time1)

    else:
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
        if cost[0]>0:
            print(f"price of slot 1 is {cost[0]}")
        cost[0] = 0
    if a2 == 1:

        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2)

        current_time2 = time.time()
        true_time2 += current_time2 - start_time
        #start_time = current_time2

        cost[1] = price(space, true_time2)
    else:
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)
        if cost[1]>0:
            print(f"price of slot 2 is {cost[1]}")
        cost[1]=0
    if a3 == 1:
        cv2.polylines(frame, [np.array(area3, np.int32)], True, (0, 0, 255), 2)

        current_time3 = time.time()
        true_time3 += current_time3 - start_time
        start_time = current_time3
        cost[2] = price(space, true_time3)

    else:
        cv2.polylines(frame, [np.array(area3, np.int32)], True, (0, 255, 0), 2)
        if cost[2]>0:
            print(f"price of slot 3 is {cost[2]}")
        cost[2]=0
    if a4 == 1:
        cv2.polylines(frame, [np.array(area4, np.int32)], True, (0, 0, 255), 2)

        current_time4 = time.time()
        true_time4 += current_time4 - start_time
        start_time = current_time4
        cost[3] = price(space, true_time4)

    else:
        cv2.polylines(frame, [np.array(area4, np.int32)], True, (0, 255, 0), 2)
        if cost[3]>0:
            print(f"price of slot 4 is {cost[3]}")
        cost[3]=0
    if a5 == 1:
        cv2.polylines(frame, [np.array(area5, np.int32)], True, (0, 0, 255), 2)

        current_time5 = time.time()
        true_time5 += current_time5 - start_time
        #start_time = current_time5
        cost[4] = price(space, true_time5)

    else:
        cv2.polylines(frame, [np.array(area5, np.int32)], True, (0, 255, 0), 2)
        if cost[4]>0:
            print(f"price of slot 5 is {cost[4]}")
        cost[4]=0
    if a6 == 1:
        cv2.polylines(frame, [np.array(area6, np.int32)], True, (0, 0, 255), 2)

        current_time6 = time.time()
        true_time6 += current_time6 - start_time
        #start_time = current_time6
        cost[5] = price(space, true_time6)

    else:
        cv2.polylines(frame, [np.array(area6, np.int32)], True, (0, 255, 0), 2)
        if cost[5]>0:
            print(f"price of slot 6 is {cost[5]}")
        cost[5]=0
    if a7 == 1:
        cv2.polylines(frame, [np.array(area7, np.int32)], True, (0, 0, 255), 2)

        current_time7 = time.time()
        true_time7 += current_time7 - start_time
        start_time = current_time7
        cost[6] = price(space, true_time7)

    else:
        cv2.polylines(frame, [np.array(area7, np.int32)], True, (0, 255, 0), 2)
        if cost[6]>0:
            print(f"price of slot 7 is {cost[6]}")
        cost[6]=0
    if a8 == 1:
        cv2.polylines(frame, [np.array(area8, np.int32)], True, (0, 0, 255), 2)

        current_time8 = time.time()
        true_time8 += current_time8 - start_time
        start_time = current_time8
        cost[7] = price(space, true_time8)
    else:
        cv2.polylines(frame, [np.array(area8, np.int32)], True, (0, 255, 0), 2)
        if cost[7]>0:
            print(f"price of slot 8 is {cost[7]}")
        cost[7]=0
    if a9 == 1:
        cv2.polylines(frame, [np.array(area9, np.int32)], True, (0, 0, 255), 2)

        current_time9 = time.time()
        true_time9 += current_time9 - start_time
        start_time = current_time9
        cost[8] = price(space, true_time9)

    else:
        cv2.polylines(frame, [np.array(area9, np.int32)], True, (0, 255, 0), 2)
        if cost[8]>0:
            print(f"price of slot 9 is {cost[8]}")
        cost[8]=0
    if a10 == 1:
        cv2.polylines(frame, [np.array(area10, np.int32)], True, (0, 0, 255), 2)

        current_time10 = time.time()
        true_time10 += current_time10 - start_time
        start_time = current_time10
        cost[9] = price(space, true_time10)

    else:
        cv2.polylines(frame, [np.array(area10, np.int32)], True, (0, 255, 0), 2)
        if cost[9]>0:
            print(f"price of slot 10 is {cost[9]}")
        cost[9]=0
    if a11 == 1:
        cv2.polylines(frame, [np.array(area11, np.int32)], True, (0, 0, 255), 2)

        current_time11 = time.time()
        true_time11 += current_time11 - start_time
        start_time = current_time11
        cost[10] = price(space, true_time11)

    else:
        cv2.polylines(frame, [np.array(area11, np.int32)], True, (0, 255, 0), 2)
        if cost[10]>0:
            print(f"price of slot 11 is {cost[10]}")
        cost[10]=0
    if a12 == 1:
        cv2.polylines(frame, [np.array(area12, np.int32)], True, (0, 0, 255), 2)

        current_time12 = time.time()
        true_time12 += current_time12 - start_time
        start_time = current_time12
        cost[11] = price(space, true_time12)
    else:
        cv2.polylines(frame, [np.array(area12, np.int32)], True, (0, 255, 0), 2)
        if cost[11]>0:
            print(f"price of slot 12 is {cost[11]}")
        cost[11]=0
    if a13 == 1:
        cv2.polylines(frame, [np.array(area13, np.int32)], True, (0, 0, 255), 2)
        #cost[12] = price(space, true_time10)

    else:
        cv2.polylines(frame, [np.array(area13, np.int32)], True, (0, 255, 0), 2)

    if a14 == 1:
        cv2.polylines(frame, [np.array(area14, np.int32)], True, (0, 0, 255), 2)



    else:
        cv2.polylines(frame, [np.array(area14, np.int32)], True, (0, 255, 0), 2)

    if a15 == 1:
        cv2.polylines(frame, [np.array(area15, np.int32)], True, (0, 0, 255), 2)

        current_time15 = time.time()
        true_time15 += current_time15 - start_time
        start_time = current_time15
        cost[14] = price(space, true_time15)
    else:
        cv2.polylines(frame, [np.array(area15, np.int32)], True, (0, 255, 0), 2)
        if cost[11]>0:
            print(f"price of slot 15 is {cost[14]}")
        cost[14]=0

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
