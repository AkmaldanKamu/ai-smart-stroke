import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def detect_facial_droop_from_frame(frame):
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return "Wajah Tidak Terdeteksi"

        face = result.multi_face_landmarks[0]
        left_mouth = face.landmark[61]
        right_mouth = face.landmark[291]

        delta_y = abs(left_mouth.y - right_mouth.y)

        if delta_y > 0.03:
            return "Miring"
        else:
            return "Normal"
