# backend/detection/voice_detection.py

import whisper
import os
import tempfile
import sounddevice as sd
import scipy.io.wavfile


# Load model Whisper hanya sekali
model = whisper.load_model("tiny")  # dari 'base' ganti ke 'tiny'

def record_audio(duration=5, sample_rate=16000, device_index=2):
    try:
        print("▶️ Merekam suara...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=device_index)
        sd.wait()

        volume_level = float(audio.max())
        print(f"🎚️ Volume Max (debug): {volume_level:.4f}")
        if volume_level < 0.01:
            print("⚠️ Suara terlalu kecil.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            scipy.io.wavfile.write(f.name, sample_rate, audio)
            return f.name
    except Exception as e:
        print(f"❌ ERROR saat merekam: {e}")
        return None

def is_speech_clear(text):
    if not text.strip():
        return "Tidak ada suara"

    text = text.lower()
    count = 0
    if "makan" in text:
        count += 1
    if "nasi" in text:
        count += 1
    if "sedang makan" in text or "saya makan" in text:
        count += 1

    if count >= 2:
        return "Suara jelas"
    elif count == 1:
        return "Suara tidak jelas"
    else:
        return "Tidak ada suara"


def detect_speech_clarity(return_text=False):
    audio_file = record_audio()
    if audio_file is None:
        return ("Gagal merekam suara", "", 2) if return_text else ("Gagal merekam suara", 2)

    try:
        result = model.transcribe(audio_file, language='id')
        text = result.get("text", "").strip()
        print(f"🗣️ Transkrip: {text}")
        os.remove(audio_file)

        if not text:
            return ("Tidak ada suara", "", 2) if return_text else ("Tidak ada suara", 2)

        # Validasi berdasarkan kata kunci
        text_lower = text.lower()
        if all(word in text_lower for word in ["makan", "nasi"]):
            return ("Suara jelas", text, 0) if return_text else ("Suara jelas", 0)
        else:
            return ("Suara tidak jelas", text, 1) if return_text else ("Suara tidak jelas", 1)

    except Exception as e:
        print(f"❌ Error Whisper: {e}")
        return (f"Error analisis suara: {e}", "", 2) if return_text else (f"Error analisis suara: {e}", 2)