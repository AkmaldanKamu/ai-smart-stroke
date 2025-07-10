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

    def detect_arm_drift(self, duration=5, show_visual=True):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {"status": "error", "message": "Kamera tidak bisa diakses."}

        start_time = time.time()
        y_deltas = []

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()

            input_image = cv2.resize(frame, (456, 256))
            input_image = input_image.transpose((2, 0, 1))
            input_image = np.expand_dims(input_image, axis=0)

            results = self.compiled_model([input_image])[self.output_layer]
            keypoints = results.reshape(-1, 3)

            left_wrist = keypoints[9]
            right_wrist = keypoints[10]

            if left_wrist[2] > 0.1 and right_wrist[2] > 0.1:
                h, w = frame.shape[:2]
                lx, ly = int(left_wrist[0] * w), int(left_wrist[1] * h)
                rx, ry = int(right_wrist[0] * w), int(right_wrist[1] * h)

                cv2.circle(display_frame, (lx, ly), 8, (0, 255, 0), -1)
                cv2.circle(display_frame, (rx, ry), 8, (0, 255, 0), -1)
                cv2.line(display_frame, (lx, ly), (rx, ry), (0, 255, 255), 2)

                delta_y = abs(left_wrist[1] - right_wrist[1])
                y_deltas.append(delta_y)

            if show_visual:
                cv2.putText(display_frame, f"Deteksi tangan selama {int(duration)} detik", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("AI-SMART | Pose Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
                break

        cap.release()
        cv2.destroyAllWindows()

        if not y_deltas:
            return {
                "status": "ok",
                "kategori": "Tidak merespons",
                "delta_y": None,
                "saran": "Tidak ada gerakan tangan terdeteksi. Harap ulangi atau pasien tidak merespons."
            }

        avg_delta = np.mean(y_deltas)

        if avg_delta < 0.03:
            kategori = "Normal"
            saran = "Gerakan tangan seimbang, tidak menunjukkan gejala stroke."
        elif avg_delta < 0.07:
            kategori = "Ringan"
            saran = "Terdapat sedikit perbedaan posisi tangan. Lanjutkan observasi."
        elif avg_delta < 0.12:
            kategori = "Sedang"
            saran = "Perbedaan tangan cukup terlihat. Konsultasikan ke medis."
        else:
            kategori = "Berat"
            saran = "Salah satu tangan turun signifikan. Kemungkinan besar stroke."

        return {
            "status": "ok",
            "kategori": kategori,
            "delta_y": round(avg_delta, 4),
            "saran": saran
        }
