import sys
import os
import cv2

# Tambahkan path ke folder backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from flask import Flask, render_template, request, jsonify
from detection.voice_detection import detect_speech_clarity
from detection.face_detection import detect_facial_droop_from_frame
from detection.hand_detection import detect_arm_drift_from_frame
from detection.voice_detection import detect_speech_clarity
from detection.nihss_scoring import score_nihss
from detection.nihss_scoring import generate_diagnosis_summary


app = Flask(__name__, template_folder="templates", static_folder="static")

# Route utama untuk halaman web
@app.route('/')
def index():
    return render_template("index.html")

# Route API untuk deteksi suara
@app.route('/analyze-audio', methods=['POST'])
def analyze_audio():
    clarity, raw_text = detect_speech_clarity(return_text=True)
    return jsonify({
        'status': 'ok',
        'hasil': clarity,
        'transkrip': raw_text
    })
    
@app.route('/diagnosa', methods=['POST'])
def diagnosa():
    # Ambil frame dari webcam
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'status': 'error', 'message': 'Gagal mengambil gambar dari kamera.'}), 500

    face_result = detect_facial_droop_from_frame(frame)
    hand_result = detect_arm_drift_from_frame(frame)
    voice_result = detect_speech_clarity()

    print(f"[Diagnosa] Wajah: {face_result} | Tangan: {hand_result} | Suara: {voice_result}")

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
        'summary': summary
    })
    
    
# Menjalankan server
if __name__ == '__main__':
    app.run(debug=True)
