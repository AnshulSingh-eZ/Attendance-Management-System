import cv2
import queue
import threading
from keras_facenet import FaceNet

model = FaceNet()

class CameraStream:
    def __init__(self, src=1):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = queue.Queue(maxsize=1)
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        return self

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)
        self.cap.release()

    def read(self):
        return self.q.get()

    def stop(self):
        self.running = False
        
def detect_faces():
    cap = CameraStream().start()
    try:
        while True:
            frame = cap.read()
            display_frame = frame.copy()
            detections = model.extract(display_frame, threshold=0.8)
            for det in detections:
                x, y, w, h = det['box']
                x = max(0, x)
                y = max(0, y)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("temp", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.stop()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    detect_faces()
