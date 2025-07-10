# backend/detection/hand_detection.py

import cv2
import numpy as np
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class StrokeDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    def is_hand_raised(self, wrist_y, shoulder_y, elbow_y, frame_height, threshold_ratio=0.15):
        arm_length = abs(shoulder_y - elbow_y)
        threshold = shoulder_y - (arm_length * threshold_ratio)
        return wrist_y < threshold

    def analyze_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        h = frame.shape[0]

        # Landmark koordinat
        l_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
        r_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
        l_elbow_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h
        r_elbow_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h
        l_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h
        r_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h

        left_raised = self.is_hand_raised(l_wrist_y, l_shoulder_y, l_elbow_y, h)
        right_raised = self.is_hand_raised(r_wrist_y, r_shoulder_y, r_elbow_y, h)

        return {
            'left_raised': left_raised,
            'right_raised': right_raised,
            'l_wrist_y': l_wrist_y,
            'r_wrist_y': r_wrist_y
        }

    def detect_static_frame(self, frame):
        result = self.analyze_frame(frame)

        if result is None:
            return {
                "status": "ok",
                "kategori": "Tidak merespons",
                "saran": "❗ Tidak ada landmark tubuh terdeteksi. Pastikan kamera menangkap tubuh dengan jelas."
            }

        left_raised = result['left_raised']
        right_raised = result['right_raised']

        if left_raised and right_raised:
            return {
                "status": "ok",
                "kategori": "Normal",
                "saran": "✅ Kedua tangan terangkat sejajar."
            }
        elif left_raised and not right_raised:
            return {
                "status": "ok",
                "kategori": "Tangan kanan turun",
                "saran": "⚠️ Tangan kanan tidak terangkat - kemungkinan stroke sisi kanan."
            }
        elif not left_raised and right_raised:
            return {
                "status": "ok",
                "kategori": "Tangan kiri turun",
                "saran": "⚠️ Tangan kiri tidak terangkat - kemungkinan stroke sisi kiri."
            }
        else:
            return {
                "status": "ok",
                "kategori": "Tidak merespons",
                "saran": "❗ Kedua tangan tidak terangkat - kemungkinan stroke berat."
            }

    def close(self):
        self.pose.close()

# Fungsi eksternal agar mudah diimpor dari app.py
stroke_detector = StrokeDetector()

def detect_arm_drift_mediapipe_from_frame(frame):
    return stroke_detector.detect_static_frame(frame)
