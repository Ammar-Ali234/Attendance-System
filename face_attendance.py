import cv2
from face_recognition.simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
sfr.get_embeddings("images/")


cap = cv2.videocapture(0)

while True:
    frame, ret = cap.read()
    if not ret:
        break

    face_location, face_name = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_location,face_name):
        x1, y1, x2, y2 = face_loc
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)

    
    cv2.imshow("Video",frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()