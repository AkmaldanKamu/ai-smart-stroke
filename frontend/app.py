import sys
import os
import cv2
import base64
import numpy as np
import tempfile
import subprocess
import whisper
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

# Load .env dari parent directory
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

# Tambahkan path ke backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from detection.voice_detection import detect_speech_clarity, is_speech_clear
from detection.face_detection import detect_facial_droop_from_frame
from detection.nihss_scoring import score_nihss, generate_diagnosis_summary
from detection.guidance_nlp import generate_guidance

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load Whisper model sekali saja
print("⏳ Memuat model Whisper...")
whisper_model = whisper.load_model("base")
print("✅ Model Whisper siap.")

# Gemini setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

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

# -------------------------------
# ROUTE: Halaman Utama
# -------------------------------
@app.route('/')
def index():
    return render_template("index.html")

# -------------------------------
# ROUTE: Analisis Audio (dipakai index.html)
# -------------------------------
@app.route('/analyze-audio', methods=['POST'])
def analyze_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'File audio tidak ditemukan'}), 400

        audio_file = request.files['audio']

        # Simpan file sementara
        temp_input  = os.path.join(tempfile.gettempdir(), 'input.webm')
        temp_output = os.path.join(tempfile.gettempdir(), 'output.wav')
        audio_file.save(temp_input)

        # Konversi webm → wav pakai ffmpeg
        subprocess.call(
            ['ffmpeg', '-y', '-i', temp_input, '-ar', '16000', '-ac', '1', temp_output],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if not os.path.exists(temp_output):
            return jsonify({'status': 'error', 'message': 'Konversi audio gagal. Pastikan ffmpeg terinstall.'}), 500

        # Transkripsi dengan Whisper
        result = whisper_model.transcribe(temp_output, language='id', fp16=False)
        text = result.get("text", "").strip()
        print(f"🗣️ Transkrip: {text}")

        # Cleanup
        for f in [temp_input, temp_output]:
            if os.path.exists(f):
                os.remove(f)

        if not text:
            return jsonify({'status': 'ok', 'hasil': '❌ Tidak Ada Suara', 'transkrip': '', 'score': 2})

        # Evaluasi kejelasan suara
        label, score = is_speech_clear(text, return_score=True)

        return jsonify({
            'status'   : 'ok',
            'hasil'    : label,
            'transkrip': text,
            'score'    : score
        })

    except Exception as e:
        print(f"❌ ERROR analyze_audio: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# -------------------------------
# ROUTE: Deteksi Wajah
# -------------------------------
@app.route('/detect-face', methods=['POST'])
def detect_face():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'Request tidak valid'}), 400

    image_b64 = data.get("image")
    frame = decode_base64_image(image_b64)

    if frame is None:
        return jsonify({'status': 'error', 'message': 'Frame tidak valid'}), 400

    try:
        result = detect_facial_droop_from_frame(frame, return_detail=True)
        return jsonify(result)
    except Exception as e:
        print(f"❌ ERROR detect_face: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# -------------------------------
# ROUTE: Diagnosa Stroke Gabungan
# -------------------------------
@app.route('/diagnosa', methods=['POST'])
def diagnosa():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'Request tidak valid'}), 400

    image_b64     = data.get("image")
    face_score    = data.get("face_score")
    face_kategori = data.get("face_kategori")
    voice_result  = data.get("voice_result")
    voice_score   = data.get("voice_score")

    if not all([image_b64, face_kategori, voice_result]) or face_score is None or voice_score is None:
        return jsonify({'status': 'error', 'message': 'Data tidak lengkap'}), 400

    try:
        total_score = int(face_score) + int(voice_score)

        if total_score == 0:
            kategori   = "Normal"
            saran      = "Pasien tidak menunjukkan tanda stroke."
            penanganan = "Tidak perlu tindakan khusus."
        elif total_score <= 2:
            kategori   = "Ringan"
            saran      = "Amati, bisa konsultasi dokter."
            penanganan = "Rujuk ke dokter umum atau klinik untuk evaluasi lebih lanjut."
        elif total_score <= 4:
            kategori   = "Sedang"
            saran      = "Segera ke rumah sakit."
            penanganan = "Bawa pasien ke rumah sakit terdekat secepatnya."
        else:
            kategori   = "Berat"
            saran      = "Panggil ambulans secepatnya!"
            penanganan = "Hubungi 119 dan siapkan tindakan darurat."

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
            'status'    : 'ok',
            'skor'      : total_score,
            'kategori'  : kategori,
            'saran'     : saran,
            'penanganan': penanganan,
            'summary'   : summary,
            'guidance'  : guidance,
            'face'      : face_kategori,
            'voice'     : voice_result
        })

    except Exception as e:
        print(f"❌ ERROR diagnosa: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# -------------------------------
# ROUTE: AI Guidance (Gemini Streaming)
# -------------------------------
@app.route('/ai-guidance', methods=['POST'])
def ai_guidance():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'Data tidak valid'}), 400

    if not GEMINI_API_KEY:
        return jsonify({'status': 'error', 'message': 'GEMINI_API_KEY belum di-set'}), 500

    skor          = data.get('skor', 0)
    kategori      = data.get('kategori', 'Normal')
    face_kategori = data.get('face_kategori', 'Normal')
    face_score    = data.get('face_score', 0)
    voice_result  = data.get('voice_result', '')
    voice_score   = data.get('voice_score', 0)
    penanganan    = data.get('penanganan', '')
    confidence    = data.get('confidence', None)

    prompt = f"""Kamu adalah asisten medis AI yang membantu deteksi dini stroke menggunakan aplikasi SiTANGGAP.

Berikut hasil skrining pasien:
- Skor NIHSS Total: {skor}/8
- Kategori: {kategori}
- Deteksi Wajah: {face_kategori} (skor {face_score}/6)
- Deteksi Suara: {voice_result} (skor {voice_score}/2)
- Penanganan awal: {penanganan}
{f"- Confidence deteksi wajah: {confidence}%" if confidence else ""}

Berikan respons dalam format berikut (gunakan Bahasa Indonesia yang hangat, jelas, dan mudah dipahami awam):

## 🩺 Analisis Kondisi
Jelaskan apa arti hasil ini untuk pasien/keluarga dalam 2-3 kalimat sederhana.

## ⚡ Tindakan Segera
Berikan 3-5 langkah konkret yang harus dilakukan SEKARANG, sesuai tingkat keparahan.

## 🚨 Tanda Bahaya
Sebutkan 3 tanda yang harus membuat mereka langsung ke IGD tanpa menunggu.

## 💊 Pantangan
Hal-hal yang TIDAK boleh dilakukan saat ini.

## 💬 Pesan untuk Keluarga
Satu paragraf singkat yang empatik dan menenangkan untuk keluarga pasien.

Catatan penting: Selalu ingatkan bahwa ini hanya skrining awal dan BUKAN pengganti diagnosis dokter."""

    def generate():
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    # Escape newlines untuk SSE
                    text = chunk.text.replace('\n', '\\n')
                    yield f"data: {text}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )

# -------------------------------
# RUN SERVER + NGROK
# -------------------------------
if __name__ == '__main__':
    from pyngrok import ngrok

    # Pastikan authtoken sudah di-set sebelumnya via:
    # ngrok config add-authtoken TOKEN_KAMU
    public_url = ngrok.connect(5000)
    print(f"\n🌐 Public URL (HTTPS): {public_url}\n")

    # use_reloader=False wajib agar ngrok tidak dobel tunnel
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)