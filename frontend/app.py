import sys
import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify

# Tambahkan path ke backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from detection.voice_detection import detect_speech_clarity
from detection.face_detection import detect_facial_droop_from_frame
from detection.hand_detection import StrokeDetector
from detection.nihss_scoring import score_nihss, generate_diagnosis_summary

detector = StrokeDetector()
app = Flask(__name__, template_folder="templates", static_folder="static")

# --------------------------
# UTIL FUNCTIONS
# --------------------------
def decode_base64_image(base64_string):
    try:
        header, encoded = base64_string.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[❌] Gagal decode base64: {e}")
        return None

def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

# -------------------------------
# ROUTE: Halaman Utama
# -------------------------------
@app.route('/')
def index():
    return render_template("index.html")


# -------------------------------
# ROUTE: Deteksi Suara
# -------------------------------
@app.route('/analyze-audio', methods=['POST'])
def analyze_audio():
    try:
        clarity, raw_text = detect_speech_clarity(return_text=True)
        return jsonify({
            'status': 'ok',
            'hasil': clarity,
            'transkrip': raw_text
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# -------------------------------
# ROUTE: Deteksi Wajah
# -------------------------------
@app.route('/detect-face', methods=['POST'])
def detect_face():
    data = request.get_json()
    image_b64 = data.get("image")
    frame = decode_base64_image(image_b64)

    if frame is None:
        return jsonify({'status': 'error', 'message': 'Frame tidak valid'}), 400

    try:
        result = detect_facial_droop_from_frame(frame, return_detail=True)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# -------------------------------
# ROUTE: Deteksi Tangan
# -------------------------------
@app.route('/detect-hand', methods=['POST'])
def detect_hand():
    data = request.get_json()
    image_b64 = data.get("image")
    frame = decode_base64_image(image_b64)

    if frame is None:
        return jsonify({'status': 'error', 'message': 'Frame tidak valid'}), 400

    result = detector.detect_from_single_frame(frame)

    return jsonify(result)


# -------------------------------
# ROUTE: Diagnosa Stroke Gabungan
# -------------------------------
@app.route('/diagnosa', methods=['POST'])
def diagnosa():
    data = request.get_json()
    image_b64 = data.get("image")

    if not data or not image_b64:
        return jsonify({'status': 'error', 'message': 'Gambar tidak ditemukan'}), 400

    frame = decode_base64_image(image_b64)
    if frame is None:
        return jsonify({'status': 'error', 'message': 'Gagal memproses gambar'}), 500

    try:
        # Deteksi wajah
        face_result_detail = detect_facial_droop_from_frame(frame, return_detail=True)
        face_result = face_result_detail.get('kategori', 'Tidak diketahui')

        # Deteksi tangan
        hand_result_obj = detector.detect_from_single_frame(frame)
        hand_result = hand_result_obj.get('kategori', 'Tidak diketahui')

        # Deteksi suara
        voice_result, _ = detect_speech_clarity(return_text=True)

        # Skoring NIHSS
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

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
