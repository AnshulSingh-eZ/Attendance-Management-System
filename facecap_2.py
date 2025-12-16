import cv2
from keras_facenet import FaceNet

cap = cv2.VideoCapture(0)
embedder = FaceNet()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = embedder.extract(rgb_frame, threshold=0.8)

    for det in detections:
        x, y, w, h = det['box']
        x, y = max(0, x), max(0, y)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
