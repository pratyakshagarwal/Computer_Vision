from ultralytics import YOLO
import cv2
import cvzone
import math

# for the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# for the videos
# cap = cv2.VideoCapture(r'Object_Detection\projects_using_yolo\construction_site_safety\ppe-3-1.mp4')

model = YOLO('Object_Detection\projects_using_yolo\construction_site_safety\ppe.pt')

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
mycolor = (0, 0, 255)
safety = ['Hardhat', 'Mask', 'Safety Vest']
not_safety = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
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
              currentClass = classNames[cls]
              print(currentClass)
              if conf > 0.45:
                   if currentClass in not_safety:
                        mycolor = (0, 0, 255)
                   elif currentClass in safety:
                        mycolor = (0, 255, 0)
                   else :
                        mycolor = (255, 0, 0)
              
                   cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=mycolor,
                                   colorT=(255,255,255),colorR=mycolor, offset=5)
                   cv2.rectangle(img, (x1, y1), (x2, y2), mycolor, 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)