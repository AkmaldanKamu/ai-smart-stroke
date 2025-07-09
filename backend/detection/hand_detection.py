import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def detect_arm_drift_from_frame(frame):
    with mp_pose.Pose(static_image_mode=False) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if not result.pose_landmarks:
            return "Tidak Terdeteksi"

        landmarks = result.pose_landmarks.landmark

        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Bandingkan tinggi pergelangan tangan dengan bahu (semakin tinggi, semakin kecil nilai y)
        left_hand_up = left_wrist.y < left_shoulder.y
        right_hand_up = right_wrist.y < right_shoulder.y

        if left_hand_up and right_hand_up:
            return "Kedua tangan naik"
        elif left_hand_up:
            return "Tangan kanan turun"
        elif right_hand_up:
            return "Tangan kiri turun"
        else:
            return "Kedua tangan turun"
