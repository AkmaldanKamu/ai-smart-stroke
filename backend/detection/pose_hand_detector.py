import cv2
import numpy as np
import time
from openvino.runtime import Core
from smart_camera_selector import find_real_camera

class PoseHandDetector:
    def __init__(self, model_path):
        try:
            ie = Core()
            self.model = ie.read_model(model=f"{model_path}.xml")
            self.compiled_model = ie.compile_model(self.model, device_name="CPU")
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
        except Exception as e:
            print(f"[ERROR] Gagal load model hand detection: {e}")

    def preprocess(self, frame):
        """
        Resize & normalize frame sesuai input model (NCHW).
        """
        input_shape = self.input_layer.shape  # [1, 3, H, W]
        _, _, h, w = input_shape
        resized = cv2.resize(frame, (w, h))
        transposed = resized.transpose((2, 0, 1))  # HWC → CHW
        input_tensor = np.expand_dims(transposed, axis=0).astype(np.float32)
        return input_tensor

    def extract_keypoints(self, heatmaps, frame_shape):
        """
        Ekstrak koordinat keypoint (x, y, confidence) dari heatmap.
        """
        keypoints = []
        num_keypoints = heatmaps.shape[1]
        heatmap_h, heatmap_w = heatmaps.shape[2], heatmaps.shape[3]
        orig_h, orig_w = frame_shape[:2]

        for i in range(num_keypoints):
            heatmap = heatmaps[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatmap)
            x = int(point[0] * orig_w / heatmap_w)
            y = int(point[1] * orig_h / heatmap_h)
            keypoints.append((x, y, conf))
        return keypoints

    def detect_arm_drift(self, duration=5):
        """
        Deteksi pergerakan tangan (lengan kiri dan kanan) selama 5 detik.
        Klasifikasi ke Stroke Berat / Ringan / Normal berdasarkan pergerakan tangan.
        """
        cap = cv2.VideoCapture(find_real_camera())
        if not cap.isOpened():
            return {"status": "error", "message": "Kamera tidak tersedia."}

        wrist_left_y = []
        wrist_right_y = []
        start_time = time.time()

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue

            input_tensor = self.preprocess(frame)
            result = self.compiled_model([input_tensor])[self.output_layer]
            keypoints = self.extract_keypoints(result, frame.shape)

            if len(keypoints) < 8:
                continue

            # COCO keypoint: 4 = R wrist, 7 = L wrist
            wrist_left_y.append(keypoints[7][1])
            wrist_right_y.append(keypoints[4][1])

        cap.release()

        if len(wrist_left_y) < 2 or len(wrist_right_y) < 2:
            return {
                "status": "ok",
                "kategori": "Tidak merespons",
                "delta_y": None,
                "saran": "Tidak ada gerakan tangan terdeteksi. Harap ulangi atau pasien tidak merespons."
            }

        delta_left = wrist_left_y[-1] - wrist_left_y[0]
        delta_right = wrist_right_y[-1] - wrist_right_y[0]
        max_delta = max(abs(delta_left), abs(delta_right))

        result = {
            "status": "ok",
            "delta_y": round(max_delta, 2)
        }

        threshold = 50  # px

        if abs(delta_left) > threshold and abs(delta_right) > threshold:
            result["kategori"] = "Normal"
            result["saran"] = "Kedua tangan terangkat sejajar dengan stabil."
        elif abs(delta_left) > threshold or abs(delta_right) > threshold:
            result["kategori"] = "Stroke Ringan"
            result["saran"] = "Satu tangan terdeteksi turun dari posisi sejajar."
        else:
            result["kategori"] = "Stroke Berat"
            result["saran"] = "Tidak ada pergerakan signifikan atau kedua tangan turun."

        return result
