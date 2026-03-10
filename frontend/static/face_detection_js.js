/**
 * face_detection_js.js
 * Port dari analyze_symmetry_pro.py ke JavaScript
 * Menggunakan MediaPipe Tasks Vision (browser-native, tanpa Python backend)
 *
 * Ekspor ke window.analyzeSymmetryJS(videoElement) → Promise<result>
 */

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

// ─── Singleton landmarker (inisialisasi sekali) ───────────────────────────────
let faceLandmarker = null;
let isLoading = false;

async function getLandmarker() {
  if (faceLandmarker) return faceLandmarker;
  if (isLoading) {
    // Tunggu sampai selesai loading
    while (isLoading) await new Promise(r => setTimeout(r, 100));
    return faceLandmarker;
  }

  isLoading = true;
  console.log("[SiTANGGAP] Memuat model FaceLandmarker...");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "IMAGE",
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });

  isLoading = false;
  console.log("[SiTANGGAP] Model siap.");
  return faceLandmarker;
}

// ─── Helper: hitung sudut antara dua titik (degrees) ─────────────────────────
function angleBetween(p1, p2) {
  return (Math.atan2(p2.y - p1.y, p2.x - p1.x) * 180) / Math.PI;
}

// ─── Fungsi utama: port dari analyze_symmetry_pro.py ─────────────────────────
async function analyzeSymmetryJS(videoEl) {
  const landmarker = await getLandmarker();

  // Capture frame dari video ke ImageData
  const canvas = document.createElement("canvas");
  canvas.width = videoEl.videoWidth || 640;
  canvas.height = videoEl.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

  // Deteksi landmarks
  const result = landmarker.detect(canvas);

  if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
    return { status: "error", message: "Wajah tidak terdeteksi" };
  }

  const lm = result.faceLandmarks[0]; // array of {x, y, z}

  // ── BIBIR (sama persis dengan Python) ──
  const lBibir  = lm[61];
  const rBibir  = lm[291];
  const deltaBibirY = Math.abs(lBibir.y - rBibir.y);
  const ratioBibirX = Math.abs(lBibir.x - rBibir.x);

  // ── MATA ──
  const topLEye = lm[159];
  const botLEye = lm[145];
  const topREye = lm[386];
  const botREye = lm[374];

  const lEyeOpenness = Math.abs(topLEye.y - botLEye.y);
  const rEyeOpenness = Math.abs(topREye.y - botREye.y);
  const deltaMata    = Math.abs(lEyeOpenness - rEyeOpenness);
  const ptosis       = deltaMata > 0.015;

  // ── PIPI / HEAD TILT ──
  const lPipi   = lm[234];
  const rPipi   = lm[454];
  const angleFace = angleBetween(lPipi, rPipi);

  // ── SCORING (sama dengan Python) ──
  const nilaiBibir = deltaBibirY < 0.01 ? 0 : deltaBibirY < 0.03 ? 1 : 2;
  const nilaiMata  = deltaMata   < 0.01 ? 0 : deltaMata   < 0.02 ? 1 : 2;
  const nilaiPipi  = Math.abs(angleFace) < 3 ? 0 : Math.abs(angleFace) < 6 ? 1 : 2;

  const nilaiTotal = nilaiBibir + nilaiMata + nilaiPipi;

  // ── KATEGORI ──
  let kategori, saran;
  if      (nilaiTotal <= 1) { kategori = "Normal"; saran = "Wajah simetris. Tidak ada tanda stroke."; }
  else if (nilaiTotal <= 3) { kategori = "Ringan"; saran = "Terdapat sedikit asimetri. Perlu observasi."; }
  else if (nilaiTotal <= 5) { kategori = "Sedang"; saran = "Asimetri wajah terlihat. Konsultasikan ke dokter."; }
  else                       { kategori = "Berat";  saran = "Tanda jelas facial droop. Segera periksa medis."; }

  return {
    status: "ok",
    kategori,
    saran,
    score_total: nilaiTotal,
    penilaian: {
      bibir: {
        delta_y: deltaBibirY.toFixed(4),
        rasio_x: ratioBibirX.toFixed(4),
        nilai: nilaiBibir,
      },
      mata: {
        delta_openness: deltaMata.toFixed(4),
        ptosis_suspect: ptosis,
        nilai: nilaiMata,
      },
      pipi: {
        kemiringan_derajat: angleFace.toFixed(2),
        nilai: nilaiPipi,
      },
    },
  };
}

// ─── Expose ke window agar bisa dipanggil dari script non-module ──────────────
window.analyzeSymmetryJS = analyzeSymmetryJS;

// Pre-load model saat halaman pertama dibuka (agar deteksi pertama lebih cepat)
getLandmarker().catch(err =>
  console.warn("[SiTANGGAP] Gagal pre-load model:", err)
);