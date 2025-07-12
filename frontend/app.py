import sys
import os
import cv2
import base64
import numpy as np
import tempfile
import subprocess
import whisper
from flask import Flask, render_template, request, jsonify


# Tambahkan path ke backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from detection.voice_detection import detect_speech_clarity, is_speech_clear
from detection.voice_detection import detect_speech_clarity
from detection.face_detection import detect_facial_droop_from_frame
from detection.nihss_scoring import score_nihss, generate_diagnosis_summary
from detection.guidance_nlp import generate_guidance

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

@app.route('/analyze-audio', methods=['POST'])
def analyze_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'File audio tidak ditemukan'}), 400

        audio_file = request.files['audio']

        # Simpan file sementara
        temp_input = os.path.join(tempfile.gettempdir(), 'input.webm')
        temp_output = os.path.join(tempfile.gettempdir(), 'output.wav')
        audio_file.save(temp_input)

        # Konversi dari webm ke wav (dengan ffmpeg)
        subprocess.call(['ffmpeg', '-y', '-i', temp_input, temp_output],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Transkripsi dengan Whisper
        model = whisper.load_model("base")
        result = model.transcribe(temp_output, language='id')
        text = result.get("text", "").strip()

        # ✅ Panggil fungsi dengan return_score=True
        label, score = is_speech_clear(text, return_score=True)

        return jsonify({
            'status': 'ok',
            'hasil': label,      # contoh: "⚠️ Suara Kurang Jelas"
            'transkrip': text,
            'score': score       # contoh: 1
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
# ROUTE: Diagnosa Stroke Gabungan
# -------------------------------
@app.route('/diagnosa', methods=['POST'])
def diagnosa():
    data = request.get_json()
    image_b64 = data.get("image")
    face_score = data.get("face_score")
    face_kategori = data.get("face_kategori")
    voice_result = data.get("voice_result")
    voice_score = data.get("voice_score")

    if not all([image_b64, face_kategori, voice_result]) or face_score is None or voice_score is None:
        return jsonify({'status': 'error', 'message': 'Data tidak lengkap'}), 400

    try:
        total_score = int(face_score) + int(voice_score)
        

        # Diagnosa Berdasarkan Skor NIHSS
        if total_score == 0:
            kategori = "Normal"
            saran = "Pasien tidak menunjukkan tanda stroke."
            penanganan = "Tidak perlu tindakan khusus."
        elif total_score <= 2:
            kategori = "Ringan"
            saran = "Amati, bisa konsultasi dokter."
            penanganan = "Rujuk ke dokter umum atau klinik untuk evaluasi lebih lanjut."
        elif total_score <= 4:
            kategori = "Sedang"
            saran = "Segera ke rumah sakit."
            penanganan = "Bawa pasien ke rumah sakit terdekat secepatnya."
        else:
            kategori = "Berat"
            saran = "Panggil ambulans secepatnya!"
            penanganan = "Hubungi 119 dan siapkan tindakan darurat."

        # Summary ringkasan diagnosis
        summary = []

        if face_kategori.lower() == "normal":
            summary.append("Tidak ditemukan kelainan pada otot wajah.")
        else:
            summary.append(f"Wajah menunjukkan gejala: {face_kategori}.")

        voice_lower = voice_result.lower()
        if "tidak ada suara" in voice_lower:
            summary.append("Pasien tidak dapat berbicara saat diminta, gejala afasia berat.")
        elif "tidak jelas" in voice_lower:
            summary.append("Ucapan terdengar tidak jelas, indikasi gangguan bicara (dysarthria).")
        else:
            summary.append("Ucapan terdengar normal.")

        summary.append(f"Kategori stroke: {kategori} ({total_score} poin).")
        summary.append(f"Saran tindakan: {saran}")
        summary.append(f"🩺 Penanganan: {penanganan}")
        guidance = generate_guidance(total_score)
        

        return jsonify({
            'status': 'ok',
            'skor': total_score,
            'kategori': kategori,
            'saran': saran,
            'penanganan': penanganan,
            'summary': summary,
            'guidance': guidance,
            'face': face_kategori,
            'voice': voice_result
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    try:
        file = request.files['audio']
        if not file:
            return jsonify({'status': 'error', 'message': 'File audio kosong'}), 400

        # Simpan file sementara
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        file.save(temp_file.name)

        # Konversi webm -> wav dengan ffmpeg
        wav_path = temp_file.name.replace(".webm", ".wav")
        os.system(f"ffmpeg -i {temp_file.name} -ar 16000 -ac 1 -y {wav_path}")

        # Transkripsi dengan Whisper
        result = model.transcribe(wav_path, language='id')
        text = result.get("text", "").strip()
        print(f"🗣️ Transkrip: {text}")

        if not text:
            return jsonify({'status': 'ok', 'hasil': "Tidak ada suara", 'transkrip': "", 'score': 2})

        text_lower = text.lower()
        if all(word in text_lower for word in ["makan", "nasi"]):
            return jsonify({'status': 'ok', 'hasil': "Suara jelas", 'transkrip': text, 'score': 0})
        else:
            return jsonify({'status': 'ok', 'hasil': "Suara tidak jelas", 'transkrip': text, 'score': 1})

    except Exception as e:
        print("❌ ERROR saat upload_audio:", e)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# -------------------------------
# RUN SERVER
# -------------------------------
# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)