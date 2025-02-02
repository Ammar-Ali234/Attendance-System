from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("Face_Model.pt")
image = cv2.imread("images/test_image.jpg")
image = cv2.resize(image, (640, 480))

class_names = ["face"]

results = model(image)
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        confidence = box.conf[0]  
        label_in = int(box.cls[0])
        
        label = class_names[label_in]
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        
        label_text = f"{label} {confidence:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()