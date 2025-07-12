import whisper
import os
import tempfile
import sounddevice as sd
import scipy.io.wavfile
import numpy as np
from flask import request
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model Whisper dengan error handling
try:
    model = whisper.load_model("tiny")
    logger.info("Model Whisper berhasil dimuat")
except Exception as e:
    logger.error(f"Gagal memuat model Whisper: {e}")
    model = None

def record_audio(duration=5, sample_rate=16000, device_index=None):
    """
    Merekam suara dari microphone dengan error handling yang lebih baik
    """
    logger.info("Memulai perekaman suara...")
    
    try:
        # Auto-detect device jika tidak dispesifikasi
        if device_index is None:
            devices = sd.query_devices()
            device_index = sd.default.device[0]  # Input device default
        
        # Rekam audio
        audio = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            device=device_index,
            dtype=np.float32
        )
        sd.wait()

        # Validasi level audio
        volume_level = float(np.max(np.abs(audio)))
        logger.info(f"Volume maksimum: {volume_level:.4f}")
        
        if volume_level < 0.005:
            logger.warning("Volume audio sangat rendah")
            return None

        # Normalisasi audio
        if volume_level > 0:
            audio = audio / volume_level * 0.8

        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            # Konversi ke int16 untuk kompatibilitas
            audio_int16 = (audio * 32767).astype(np.int16)
            scipy.io.wavfile.write(f.name, sample_rate, audio_int16)
            logger.info(f"Audio disimpan: {f.name}")
            return f.name
            
    except Exception as e:
        logger.error(f"Error saat merekam: {e}")
        return None

def process_uploaded_audio(audio_file):
    """
    Proses file audio yang diupload
    """
    try:
        # Validasi file
        if not os.path.exists(audio_file.filename):
            return None
            
        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            audio_file.save(f.name)
            return f.name
            
    except Exception as e:
        logger.error(f"Error processing uploaded audio: {e}")
        return None

def is_speech_clear(text, return_score=False):
    """
    Evaluasi kejelasan suara berdasarkan jumlah kata dari kalimat target:
    'Hari ini saya makan nasi'

    Skor NIHSS:
    - 5 kata cocok → ✅ Suara Jelas (0)
    - 4 kata cocok → ✅ Suara Hampir Jelas (0)
    - 2–3 kata     → ⚠️ Suara Kurang Jelas (1)
    - 1 kata       → ❌ Suara Tidak Jelas (2)
    - 0 kata       → ❌ Tidak Ada Suara (2)
    """
    if not text or not text.strip():
        return ("❌ Tidak Ada Suara", 2) if return_score else "❌ Tidak Ada Suara"

    text_lower = text.lower()

    keywords = {
        'hari': ['hari', 'harii', 'harini'],
        'ini': ['ini', 'ni', 'ne'],
        'saya': ['saya', 'sa', 'syaa', 'sya'],
        'makan': ['makan', 'mkan', 'akan', 'mkn', 'makanan', 'makana', 'mknn'],
        'nasi': ['nasi', 'nas', 'nai', 'asi']
    }

    matched_keywords = set()

    for key, variations in keywords.items():
        for variation in variations:
            if variation in text_lower:
                matched_keywords.add(key)
                break

    matched_count = len(matched_keywords)

    if matched_count == 5:
        label, nihss_score = "✅ Suara Jelas", 0
    elif matched_count == 4:
        label, nihss_score = "✅ Suara Hampir Jelas", 0
    elif 2 <= matched_count <= 3:
        label, nihss_score = "⚠️ Suara Kurang Jelas", 1
    elif matched_count == 1:
        label, nihss_score = "❌ Suara Tidak Jelas", 2
    else:
        label, nihss_score = "❌ Tidak Ada Suara", 2

    return (label, nihss_score) if return_score else label




def detect_speech_clarity(return_text=False, audio_file=None):
    """
    Deteksi kejelasan ucapan dan skor NIHSS berdasarkan hasil transkrip Whisper
    """
    if model is None:
        error_msg = "Model Whisper tidak tersedia"
        return (error_msg, "", 0) if return_text else error_msg
    
    # Gunakan file yang diupload atau rekam baru
    if audio_file is None:
        audio_file = record_audio()
    
    if audio_file is None:
        error_msg = "Gagal memproses audio"
        return (error_msg, "", 0) if return_text else error_msg

    logger.info("Memulai analisis suara...")
    
    try:
        # Transkrip menggunakan Whisper
        result = model.transcribe(
            audio_file, 
            language='id',
            fp16=False,  
            verbose=False
        )
        
        text = result.get("text", "").strip()
        logger.info(f"Transkrip: {text}")

        # Bersihkan file sementara
        if os.path.exists(audio_file):
            os.remove(audio_file)

        # ✅ Dapatkan label dan skor dari fungsi is_speech_clear
        label, skor = is_speech_clear(text, return_score=True)

        if return_text:
            return label, text, skor  # ⬅️ PENTING! Skor NIHSS yang benar
        else:
            return label

    except Exception as e:
        logger.error(f"Error dalam analisis Whisper: {e}")
        error_msg = f"Error analisis suara: {str(e)}"
        return (error_msg, "", 0) if return_text else error_msg