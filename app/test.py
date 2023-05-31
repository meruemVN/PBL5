import cv2
from time import time
import mediapipe as mp
# import pyautogui
import numpy as np
import pickle
import pandas as pd
import urllib

# Khởi tạo các đối tượng MediaPipe Holistic và mô hình phân loại
model, label_decoder = pickle.load(open('app/model.pkl', 'rb'))

# Initializing mediapipe pose class.
mp_holistic = mp.solutions.holistic

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

# Đọc video stream từ URL của esp32-cam
cap = cv2.VideoCapture('http://192.168.0.100:81/stream')

while True:
    # Đọc một khung hình từ video stream
    ret, frame = cap.read()

    if not ret:
        break

    # Chuyển đổi khung hình sang định dạng RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xác định tư thế bằng MediaPipe Holistic
    result = mp_holistic.process(frame_rgb)
    pose_landmarks = result.pose_landmarks

    if pose_landmarks is not None:
        # Chuẩn hóa các điểm landmark của tư thế
        pose_points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose_landmarks.landmark], dtype=np.float32)
        pose_points -= np.mean(pose_points, axis=0)

        # Phân loại tư thế bằng mô hình phân loại
        pose_label = label_decoder[model.predict_classes(np.expand_dims(pose_points, axis=0))[0]]

        # Hiển thị tư thế trên khung hình
        mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.putText(frame, pose_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị khung hình kết quả
    cv2.imshow('Pose Estimation', frame)
    
    # Chờ bấm phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()