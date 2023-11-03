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