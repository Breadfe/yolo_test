"""
reference1 : https://webnautes.tistory.com/1851
reference2 : https://velog.io/@junwoo0525/YOLOv8%EC%9D%84-OpenCV%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-%EB%8F%99%EC%9E%91%EC%8B%9C%ED%82%A4%EA%B8%B0
"""

import time
import datetime # fps 계산을 위함

import cv2
start = time.time()
from ultralytics import YOLO
end = time.time()
print(end-start)

CONFIDENCE_THRESHOLD = 0.6
# GREEN = (0, 255, 0)
# WHITE = (255, 255, 255)

# coco128 = open('./yolov8_pretrained/coco128.txt', 'r')
# data = coco128.read()
# print(data)
# class_list = data.split('\n')
# coco128.close()

model = YOLO('./yolov8_pretrained/yolov8n.pt')


cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        a = results[0].boxes.data.tolist()[0][4]
        print('------------------------')
        print('results = ', a)
        print('------------------------')
        if a < CONFIDENCE_THRESHOLD:
            continue
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()