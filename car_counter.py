from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture('people.mp4')

model = YOLO("yolov8n.pt")

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

mask = cv2.imread("mask.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalCountsUp = []
totalCountsDown = []

while True:
    success, img = cap.read()
    if not success:
        break

    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)

    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["person"] and conf > 0.3:
                # cvzone.putTextRect(
                #     img, f"{currentClass} - {conf}", (max(0, x1), max(35, y1)), offset=3, scale=0.6, thickness=1)
                # cvzone.cornerRect(img, bbox=(x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limitsUp[0], limitsUp[1]),
             (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)

    cv2.line(img, (limitsDown[0], limitsDown[1]),
             (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(img, bbox=(x1, y1, w, h), l=9,
                          rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(
            img, f"{int(Id)}", (max(0, x1), max(35, y1)), offset=10, scale=2, thickness=3)

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1]-15 < cy < limitsUp[3]+15 and totalCountsUp.count(Id) == 0:
            totalCountsUp.append(Id)
            cv2.line(img, (limitsUp[0], limitsUp[1]),
                     (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
        
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1]-15 < cy < limitsDown[3]+15 and totalCountsDown.count(Id) == 0:
            totalCountsDown.append(Id)
            cv2.line(img, (limitsDown[0], limitsDown[1]),
                     (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    cv2.putText(
        img, str(len(totalCountsUp)), (930, 350), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 8)

    cv2.putText(
        img, str(len(totalCountsDown)), (1180, 350), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(1)
