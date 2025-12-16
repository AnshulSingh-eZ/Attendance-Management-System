# import cv2
# path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(path)
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     print("Faces detected:", len(faces))
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     cv2.imshow("Face Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import tkinter as tk
# import os
# import pickle
# root = tk.Tk()
# root.title("AMS")
# root.geometry("500x500")
# label = tk.Label(root, text="Name : ")
# label.pack()
# root.mainloop()

# modelFile = r"D:\Documents\Anshul\Projects\Attendence Management System\res10_300x300_ssd_iter_140000.caffemodel"
# configFile = r"D:\Documents\Anshul\Projects\Attendence Management System\deploy.prototxt"
# net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# cap = cv2.VideoCapture(1)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     h, w = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#                                  (300, 300), (104.0, 177.0, 123.0))

#     net.setInput(blob)
#     detections = net.forward()

#     face_count = 0
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.6: 
#             face_count += 1
#             box = detections[0, 0, i, 3:7] * [w, h, w, h]
#             (x1, y1, x2, y2) = box.astype("int")
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     print("Faces Detected:", face_count)
#     cv2.imshow("DNN Face Detection", frame)




# import cv2
# import os
# import pickle
# import numpy as np
# from keras_facenet import FaceNet

# # Load or initialize embeddings
# if os.path.exists("embeddings.pkl"):
#     with open("embeddings.pkl", "rb") as f:
#         embeddings = pickle.load(f)
# else:
#     embeddings = {}

# embedder = FaceNet()

# modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
# configFile = "deploy.prototxt"
# net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# def get_embedding(face_pixels):
#     face_pixels = np.expand_dims(face_pixels, axis=0)
#     yhat = embedder.embeddings(face_pixels)
#     return yhat[0]

# def getUserId():
#     pathuserid = "UserIdCounter.txt"
#     if not os.path.exists(pathuserid):
#         with open(pathuserid, "w") as f:
#             f.write("1")
#         return 1
#     with open(pathuserid, "r") as f:
#         user_id = int(f.read())
#     with open(pathuserid, "w") as f:
#         f.write(str(user_id + 1))
#     return user_id

# def save_emb():
#     with open("embeddings.pkl", "wb") as f:
#         pickle.dump(embeddings, f)

# def captureImage(user_id, totalcaps=50):
#     pathsaveimg = os.path.join("RegisteredUsers", str(user_id))
#     os.makedirs(pathsaveimg, exist_ok=True)
#     cap = cv2.VideoCapture(0)
#     cnt = 0
#     while cnt < totalcaps:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         h, w = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#                                      (300, 300), (104.0, 177.0, 123.0))
#         net.setInput(blob)
#         detections = net.forward()
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > 0.8:
#                 box = detections[0, 0, i, 3:7] * [w, h, w, h]
#                 (x1, y1, x2, y2) = box.astype("int")
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(w, x2), min(h, y2)
#                 face = frame[y1:y2, x1:x2]
#                 if face.size == 0 or min(face.shape[:2]) < 50:
#                     continue
#                 cv2.imwrite(os.path.join(pathsaveimg, f"{cnt + 1}.jpg"), face)
#                 cnt += 1
#                 break
#         cv2.imshow("Registration", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

#     # Compute average embedding
#     person_embedding = []
#     for img_get in os.listdir(pathsaveimg):
#         img = cv2.imread(os.path.join(pathsaveimg, img_get))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (160, 160))
#         if cv2.Laplacian(img, cv2.CV_64F).var() < 100:
#             continue
#         img = img.astype('float32') / 255.0
#         emb = get_embedding(img)
#         if np.any(np.isnan(emb)):
#             continue
#         person_embedding.append(emb)

#     if person_embedding:
#         final_embedding = np.mean(person_embedding, axis=0)
#         embeddings[str(user_id)] = final_embedding
#         save_emb()
#         print(f"User {user_id} registered with {len(person_embedding)} valid embeddings.")
#     else:
#         print("Error: No valid embeddings found. User not registered.")

# def recognize():
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         h, w = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#                                      (300, 300), (104.0, 177.0, 123.0))
#         net.setInput(blob)
#         detections = net.forward()
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > 0.6:
#                 box = detections[0, 0, i, 3:7] * [w, h, w, h]
#                 (x1, y1, x2, y2) = box.astype("int")
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(w, x2), min(h, y2)
#                 face = frame[y1:y2, x1:x2]
#                 if face.size == 0 or min(face.shape[:2]) < 50:
#                     continue
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#                 face = cv2.resize(face, (160, 160))
#                 face = face.astype('float32') / 255.0
#                 emb = get_embedding(face)
#                 if np.any(np.isnan(emb)):
#                     continue
#                 min_dist = float("inf")
#                 best_match = "Unknown"
#                 for user_id, stored_emb in embeddings.items():
#                     if stored_emb is None or np.any(np.isnan(stored_emb)):
#                         continue
#                     dist = np.linalg.norm(emb - stored_emb)
#                     if dist < min_dist:
#                         min_dist = dist
#                         best_match = user_id
#                 if min_dist < 1.0:
#                     cv2.putText(frame, f"ID:{best_match}", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#                 else:
#                     cv2.putText(frame, "Unknown", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2),
#                               (0, 255, 0) if min_dist < 1.0 else (0, 0, 255), 2)
#         cv2.imshow("Recognition", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# # To register a new user:
# # user_id = getUserId()
# # captureImage(user_id)

# # To start recognition:
# print(embeddings)
# recognize()



