import sys
import os
import cv2

# Tambahkan path ke backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from flask import Flask, render_template, jsonify
from detection.voice_detection import detect_speech_clarity
from detection.face_detection import detect_facial_droop_from_frame
from detection.hand_detection import detect_arm_drift_openvino
from detection.nihss_scoring import score_nihss, generate_diagnosis_summary
from smart_camera_selector import find_real_camera

app = Flask(__name__, template_folder="templates", static_folder="static")

# -------------------------------
# ROUTE HALAMAN UTAMA
# -------------------------------
@app.route('/')
def index():
    return render_template("index.html")


# -------------------------------
# ROUTE ANALISIS SUARA
# -------------------------------
@app.route('/analyze-audio', methods=['POST'])
def analyze_audio():
    clarity, raw_text = detect_speech_clarity(return_text=True)
    return jsonify({
        'status': 'ok',
        'hasil': clarity,
        'transkrip': raw_text
    })


# -------------------------------
# ROUTE DETEKSI WAJAH
# -------------------------------
@app.route('/detect-face', methods=['POST'])
def detect_face():
    cap = cv2.VideoCapture(find_real_camera())
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'status': 'error', 'message': 'Gagal akses kamera'}), 500

    result = detect_facial_droop_from_frame(frame, return_detail=True)
    return jsonify(result)


# -------------------------------
# ROUTE DETEKSI TANGAN
# -------------------------------
@app.route('/detect-hand', methods=['POST'])
def detect_hand():
    result = detect_arm_drift_openvino()
    return jsonify(result)


# -------------------------------
# ROUTE DIAGNOSA KOMBINASI (Wajah + Tangan + Suara)
# -------------------------------
@app.route('/diagnosa', methods=['POST'])
def diagnosa():
    cap = cv2.VideoCapture(find_real_camera())
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'status': 'error', 'message': 'Gagal mengambil gambar dari kamera'}), 500

    # Deteksi multimodal
    face_result_detail = detect_facial_droop_from_frame(frame, return_detail=True)
    face_result = face_result_detail.get('kategori', 'Tidak diketahui')

    hand_result_obj = detect_arm_drift_openvino()
    hand_result = hand_result_obj.get('kategori', 'Tidak diketahui')

    voice_result, _ = detect_speech_clarity(return_text=True)

    # Skoring NIHSS & rekomendasi
    scoring = score_nihss(face_result, hand_result, voice_result)
    summary = generate_diagnosis_summary(face_result, hand_result, voice_result, scoring)

    return jsonify({
        'status': 'ok',
        'face': face_result,
        'hand': hand_result,
        'voice': voice_result,
        'skor': scoring['score'],
        'kategori': scoring['kategori'],
        'saran': scoring['saran'],
        'rincian': scoring['rincian'],
        'summary': summary,
        'face_details': face_result_detail,
        'hand_details': hand_result_obj
    })


# -------------------------------
# JALANKAN FLASK APP
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
