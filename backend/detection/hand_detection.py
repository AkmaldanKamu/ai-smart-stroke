# backend/detection/hand_detection.py

import cv2
import mediapipe as mp

def detect_arm_drift_from_frame(frame):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return "Tangan tidak terdeteksi"

    left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    diff = abs(left.y - right.y)
    if diff < 0.05:
        return "Kedua tangan naik"
    elif left.y > right.y:
        return "Tangan kiri turun"
    else:
        return "Tangan kanan turun"
