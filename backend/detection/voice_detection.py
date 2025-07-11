# backend/detection/voice_detection.py

import whisper
import os
import tempfile
import sounddevice as sd
import scipy.io.wavfile

# Load model Whisper hanya sekali
model = whisper.load_model("tiny")  # dari 'base' ganti ke 'tiny'

def record_audio(duration=5, sample_rate=16000, device_index=2):
    """
    Merekam suara dari microphone dan menyimpannya sebagai file .wav.
    """
    print("▶️ Merekam suara... Silakan bicara.")
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=device_index)
        sd.wait()

        volume_level = float(audio.max())
        print(f"🎚️ Volume Max (debug): {volume_level:.4f}")
        if volume_level < 0.01:
            print("⚠️ Suara terlalu kecil. Pastikan mic aktif dan dekat.")

        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            scipy.io.wavfile.write(f.name, sample_rate, audio)
            print(f"🎧 Audio disimpan di: {f.name}")
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
    """
    Merekam suara, transkrip menggunakan Whisper, dan menentukan apakah suara jelas.
    Jika return_text=True, akan mengembalikan juga teks mentah hasil transkrip.
    """
    audio_file = record_audio()
    if audio_file is None:
        return ("Gagal merekam suara", "") if return_text else "Gagal merekam suara"

    print("🔍 Menganalisis suara...")
    try:
        result = model.transcribe(audio_file, language='id')
        text = result.get("text", "").strip()
        print(f"🗣️ Transkrip: {text}")

        os.remove(audio_file)

        hasil = is_speech_clear(text)
        return (hasil, text) if return_text else hasil

    except Exception as e:
        print(f"❌ Error Whisper: {e}")
        return (f"Error analisis suara: {e}", "") if return_text else f"Error analisis suara: {e}"
