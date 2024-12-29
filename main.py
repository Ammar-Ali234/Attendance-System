from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("Attendance-System\\Face_Model.pt")
image = cv2.imread("Attendance-System\\test_image.jpg")

results = model(image)
print(results)