import os
from detection.pose_hand_detector import PoseHandDetector

# Deteksi jalur model OpenVINO
try:
    # Cari path absolut dari model pose
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        current_dir, "..", "..", "models", "openvino_models",
        "pose", "human-pose-estimation-0001", "human-pose-estimation-0001"
    )
    model_path = os.path.abspath(model_path)

    pose_detector = PoseHandDetector(model_path)
except Exception as e:
    print(f"[ERROR] Gagal load model hand detection:\n{e}")
    pose_detector = None


def detect_arm_drift_openvino():
    """
    Jalankan deteksi gerakan tangan dengan model OpenVINO.
    Return hasil kategori deteksi + saran.
    """
    if pose_detector is None:
        return {
            "status": "error",
            "message": "Model deteksi tangan tidak tersedia / gagal di-load."
        }

    return pose_detector.detect_arm_drift()
