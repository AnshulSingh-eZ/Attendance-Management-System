# import cv2
# import os
# import mysql.connector
# import pickle
# import numpy as np
# from keras_facenet import FaceNet
# import openpyxl
# from datetime import datetime
# from flask import Flask, render_template, request, Response, redirect, url_for

# app = Flask(__name__)

# conn = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="Anshul@2006",
#     database="AMS"
# )
# cursor = conn.cursor()

# CAMERA = 0
# users_path = "RegisteredUsers"
# embedding_path = "embedding.pkl"

# ## for flask-routes things ##
# registration_active = False
# _reg_name = None
# _reg_div = None
# _reg_rollno = None


# attendance_active = False
# embedder = FaceNet()
# totalimg = 30

# if os.path.exists(embedding_path):
#     with open(embedding_path, 'rb') as f:
#         embeddings = pickle.load(f)
# else:
#     embeddings = {}


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

# def mark_attendance(name, div, rollno):
#     folder = "Data"
#     os.makedirs(folder, exist_ok=True)
#     today = datetime.now().strftime("%Y-%m-%d")
#     file_path = os.path.join(folder, f"{today}.xlsx")

#     if not os.path.exists(file_path):
#         wb = openpyxl.Workbook()
#         ws = wb.active
#         ws.append(["Name", "Class", "Roll No", "Time"])
#     else:
#         wb = openpyxl.load_workbook(file_path)
#         ws = wb.active
#         for row in ws.iter_rows(min_row=2, values_only=True):
#             if row[0] == name and row[1] == div and row[2] == rollno:
#                 wb.close()
#                 return

#     time_str = datetime.now().strftime("%H:%M:%S")
#     ws.append([name, div, rollno, time_str])
#     wb.save(file_path)
#     wb.close()

# def add_students(userid, name, div, rollno):
#     query = "INSERT INTO attendance_records VALUES(%s, %s, %s, %s)"
#     cursor.execute(query, (userid, name, div, rollno))
#     conn.commit()

# def get_student(userid):
#     query = "SELECT name, class, rollno FROM attendance_records WHERE userID=%s"
#     cursor.execute(query, (userid,))
#     result = cursor.fetchone()
#     if result:
#         return result
#     return ("Unknown", "Unknown", "Unknown")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/register', methods=['POST'])
# def register():
#     global registration_active, _reg_name, _reg_div, _reg_rollno

#     _reg_name = request.form['name']
#     _reg_div = request.form['div']
#     _reg_rollno = int(request.form['rollno'])
#     registration_active = True
#     return ("", 204)

# @app.route('/video_feed_register')
# def video_feed_register():
#     return Response(gen_frames_register(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/register_complete')
# def register_complete():
#     return ("Registration complete", 200)


# def gen_frames_register():
#     global registration_active, _reg_name, _reg_div, _reg_rollno
#     if not registration_active:
#         return
#     user_id = getUserId()
#     add_students(user_id, _reg_name, _reg_div, _reg_rollno)

#     user_dir = os.path.join(users_path, str(user_id))
#     os.makedirs(user_dir, exist_ok=True)

#     cap = cv2.VideoCapture(CAMERA)
#     count = 0
#     emb_list = []

#     while count < totalimg and registration_active:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         detections = embedder.extract(frame, threshold=0.8)
#         if detections:
#             det = detections[0]
#             x, y, w, h = det['box']
#             x, y = max(0, x), max(0, y)
#             face_img = frame[y:y+h, x:x+w]
#             if face_img.size > 0:
#                 img_path = os.path.join(user_dir, f"{count}.jpg")
#                 cv2.imwrite(img_path, face_img)

#                 face_emb = embedder.embeddings([face_img])[0]
#                 emb_list.append(face_emb)
#                 count += 1
#         if detections:
#             det = detections[0]
#             x, y, w, h = det['box']
#             x, y = max(0, x), max(0, y)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, f"{count}/{totalimg}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         ret2, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()
#     registration_active = False
#     if len(emb_list) == totalimg:
#         avg_emb = np.mean(emb_list, axis=0)
#         embeddings[user_id] = avg_emb
#         with open(embedding_path, 'wb') as f:
#             pickle.dump(embeddings, f)

# @app.route('/video_feed')
# def video_feed():
#     global attendance_active
#     attendance_active = True
#     return Response(gen_frames_attendance(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def gen_frames_attendance():
#     global attendance_active
#     cap = cv2.VideoCapture(CAMERA)
#     while attendance_active:
#         success, frame = cap.read()
#         if not success:
#             break

#         detections = embedder.extract(frame, threshold=0.8)
#         if detections:
#             for det in detections:
#                 x, y, w, h = det['box']
#                 x, y = max(0, x), max(0, y)
#                 face_img = frame[y:y+h, x:x+w]
#                 if face_img.size == 0:
#                     continue

#                 face_emb = det['embedding']
#                 min_dist = float("inf")
#                 identity = None
#                 for uid, db_emb in embeddings.items():
#                     dist = np.linalg.norm(face_emb - db_emb)
#                     if dist < min_dist:
#                         min_dist = dist
#                         identity = uid

#                 if min_dist < 1.0:
#                     name_st, div_st, rollno_st = get_student(identity)
#                     label = f"{name_st} {div_st} ({rollno_st})"
#                     try:
#                         mark_attendance(name_st, div_st, rollno_st)
#                     except PermissionError:
#                         pass
#                 else:
#                     label = "Unknown"

#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x, y - 10),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()

# @app.route('/stop_attendance')
# def stop_attendance():
#     global attendance_active
#     attendance_active = False
#     return ("Attendance monitoring stopped", 200)

# if __name__ == '__main__':
#     os.makedirs(users_path, exist_ok=True)
#     os.makedirs("Data", exist_ok=True)
#     app.run(debug=True)