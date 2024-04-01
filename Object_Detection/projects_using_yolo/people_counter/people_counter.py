from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# for the videos
cap = cv2.VideoCapture('Object_Detection\projects_using_yolo\people_counter\people.mp4')

model = YOLO('yolo_weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread('Object_Detection\projects_using_yolo\people_counter\mask.png')
mask = cv2.resize(mask, (1280, 720))

# Tracking 
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# line
limits_up = [103, 161, 296, 161]
limits_dowm = [527, 489, 735, 489]


totalcounts_up = []
totalcounts_down = []

to_detect = ['person']

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    # imgRegion = img

    imgGraphics = cv2.imread("Object_Detection\projects_using_yolo\people_counter\graphics.png", cv2.IMREAD_UNCHANGED)
    imgRegion = cvzone.overlayPNG(imgRegion, imgGraphics, (730, 260))

    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in results:
         boxes = r.boxes
         for box in boxes:
              x1, y1, x2, y2 = box.xyxy[0]
          #     print(x1, y1, x2, y2)
              x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          #     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0), 3)

              w, h = x2-x1, y2-y1
          #     confidence
              conf = math.ceil(box.conf[0]*100)/100
          #     class names
              cls = int(box.cls[0])
              detected_object = classNames[cls]
              if detected_object in to_detect and conf > 0.3:
                #   cvzone.cornerRect(imgRegion, (x1,y1,w,h), l=9)
                #   cvzone.putTextRect(imgRegion, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)
                   currentArray = np.array([x1, y1, x2, y2, conf])
                   detections = np.vstack((detections, currentArray))


    resultsTracker = tracker.update(detections)
    cv2.line(imgRegion, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 0, 255), 5)
    cv2.line(imgRegion, (limits_dowm[0], limits_dowm[1]), (limits_dowm[2], limits_dowm[3]), (0, 0, 255), 5)
    for result in resultsTracker:
         x1, y1, x2, y2, id = result
         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
         print(result)
         w, h = x2 - x1, y2 - y1
         cvzone.cornerRect(imgRegion, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
         cvzone.putTextRect(imgRegion, f'{int(id)}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)

         cx, cy = x1+w//2, y1+h//2
         cv2.circle(imgRegion, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

         if limits_up[0] < cx < limits_up[2] and limits_up[1]-15 < cy < limits_up[3]+15:
              if totalcounts_up.count(id) == 0:
                  totalcounts_up.append(id)
                  cv2.line(imgRegion, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 255, 0), 5)
        
         if limits_dowm[0] < cx < limits_dowm[2] and limits_dowm[1]-15 < cy < limits_dowm[3]+15:
              if totalcounts_down.count(id) == 0:
                  totalcounts_down.append(id)
                  cv2.line(imgRegion, (limits_dowm[0], limits_dowm[1]), (limits_dowm[2], limits_dowm[3]), (0, 255, 0), 5)

        # #  cvzone.putTextRect(imgRegion, f'Count:{len(totalcounts)}', (50, 50))
         cv2.putText(imgRegion,str(len(totalcounts_up)),(929, 345),cv2.FONT_HERSHEY_PLAIN,5,(139, 195 ,175),7)
         cv2.putText(imgRegion,str(len(totalcounts_down)),(1191, 345),cv2.FONT_HERSHEY_PLAIN,5,(50, 50, 230),7)

    # cv2.imshow('Image', img)
    cv2.imshow('Mask', imgRegion)
    cv2.waitKey(1)
