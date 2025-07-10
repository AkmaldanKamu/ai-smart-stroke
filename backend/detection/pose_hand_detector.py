import cv2
import time
import numpy as np
from openvino.runtime import Core

class PoseHandDetector:
    def __init__(self, model_path):
        self.ie = Core()
        self.model = self.ie.read_model(f"{model_path}.xml")
        self.compiled_model = self.ie.compile_model(self.model, "CPU")
        self.input_layer = self.model.inputs[0]
        self.output_layer = self.compiled_model.output(0)

    def detect_arm_drift(self, duration=5):
    cap = cv2.VideoCapture(0)
    wrist_ys_left, wrist_ys_right = [], []

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        # Preprocessing + infer pose
        # Ambil koordinat wrist kiri & kanan
        # Misal:
        wrist_left_y = ... # hasil deteksi
        wrist_right_y = ...

        wrist_ys_left.append(wrist_left_y)
        wrist_ys_right.append(wrist_right_y)

    cap.release()

    # Hitung delta antara awal dan akhir
    delta_left = wrist_ys_left[-1] - wrist_ys_left[0]
    delta_right = wrist_ys_right[-1] - wrist_ys_right[0]

    if abs(delta_left) > threshold or abs(delta_right) > threshold:
        return "Terdeteksi penurunan tangan"
    else:
        return "Stabil"