import os
from detection.pose_hand_detector import PoseHandDetector

try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.path.join(project_root, "models", "openvino_models", "pose", "human-pose-estimation-0001", "human-pose-estimation-0001")
    pose_detector = PoseHandDetector(model_path)
except Exception as e:
    print(f"[ERROR] Gagal load model hand detection: {e}")
    pose_detector = None

def detect_arm_drift_openvino():
    if pose_detector is None:
        return {
            "status": "error",
            "message": "Model deteksi tangan gagal di-load"
        }
    return pose_detector.detect_arm_drift()
