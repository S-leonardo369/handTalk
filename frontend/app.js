/**
 * ASL Translator — Frontend
 * ─────────────────────────
 * Uses MediaPipe Tasks Vision (current API, not deprecated CDN packages).
 * Connects to the FastAPI backend via WebSocket for real-time predictions.
 */

import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.js";

// ── Config ────────────────────────────────────────────────────────────────────
const API_BASE     = "http://localhost:8000";
const WS_BASE      = "ws://localhost:8000";
const CLIENT_ID    = Math.random().toString(36).slice(2);
const FRAME_SIZE   = 30;   // frames per prediction window
const STEP_SIZE    = 10;   // send every N frames
const NUM_FEATURES = 63;   // 21 landmarks × 3 coords

// ── DOM refs ──────────────────────────────────────────────────────────────────
const video       = document.getElementById("video");
const canvas      = document.getElementById("overlay");
const ctx         = canvas.getContext("2d");
const cameraNotice = document.getElementById("cameraNotice");
const hudSign     = document.getElementById("hudSign");
const confBar     = document.getElementById("confBar");
const confPct     = document.getElementById("confPct");
const signBuf     = document.getElementById("signBuffer");
const sentList    = document.getElementById("sentenceList");
const glossLog    = document.getElementById("glossLog");
const modelDot    = document.getElementById("modelDot");
const modelStatus = document.getElementById("modelStatus");
const signCount   = document.getElementById("signCount");
const btnStart    = document.getElementById("btnStart");
const btnFlush    = document.getElementById("btnFlush");
const btnClear    = document.getElementById("btnClear");
const btnSpeak    = document.getElementById("btnSpeak");
const confThresh  = document.getElementById("confThresh");
const confThreshV = document.getElementById("confThreshVal");
const pauseThresh = document.getElementById("pauseThresh");
const pauseThreshV = document.getElementById("pauseThreshVal");
const toastEl     = document.getElementById("toast");

// ── State ─────────────────────────────────────────────────────────────────────
let handLandmarker = null;
let ws             = null;
let frameBuffer    = [];
let frameCount     = 0;
let isRunning      = false;
let lastSentence   = "";
let rafId          = null;

// ── Toast helper ──────────────────────────────────────────────────────────────
let toastTimer = null;
function toast(msg, duration = 2800) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), duration);
}

// ── Backend status check ──────────────────────────────────────────────────────
async function checkBackend() {
  try {
    const r = await fetch(`${API_BASE}/status`, { signal: AbortSignal.timeout(3000) });
    const d = await r.json();
    if (d.model_loaded) {
      modelDot.className    = "dot ok";
      modelStatus.textContent = "Model ready";
      signCount.textContent = `${d.num_signs} signs`;
    } else {
      modelDot.className    = "dot warn";
      modelStatus.textContent = "No model — train first";
      signCount.textContent = "0 signs";
      toast("⚠️ No trained model found. Run training/train_model.py first.", 5000);
    }
  } catch {
    modelDot.className    = "dot error";
    modelStatus.textContent = "Backend offline";
    toast("❌ Backend not reachable. Start: uvicorn main:app --reload", 5000);
  }
}

// ── WebSocket connection ──────────────────────────────────────────────────────
function connectWS() {
  if (ws && ws.readyState <= 1) return;

  ws = new WebSocket(`${WS_BASE}/ws/${CLIENT_ID}`);

  ws.onopen = () => {
    console.log("[WS] Connected");
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === "prediction") {
      const sign = msg.sign;
      const conf = msg.confidence;
      updateHUD(sign, conf);
      updateSignBuffer(msg.buffer || []);
    }

    if (msg.type === "sentence") {
      addSentence(msg.sentence, msg.gloss);
      updateSignBuffer([]);
    }

    if (msg.type === "error") {
      console.warn("[WS]", msg.message);
    }
  };

  ws.onclose = () => {
    console.log("[WS] Disconnected — will retry in 3s");
    setTimeout(connectWS, 3000);
  };

  ws.onerror = (err) => {
    console.error("[WS] Error", err);
  };
}

// ── MediaPipe initialisation ──────────────────────────────────────────────────
async function initMediaPipe() {
  toast("Loading MediaPipe…");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence:  0.50,
    minTrackingConfidence:      0.50,
  });
  toast("MediaPipe ready ✓");
}

// ── Camera start ──────────────────────────────────────────────────────────────
async function startCamera() {
  btnStart.disabled = true;
  btnStart.textContent = "Starting…";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    });
    video.srcObject = stream;
    await new Promise((res) => (video.onloadedmetadata = res));
    await video.play();
    cameraNotice.classList.add("hidden");
    btnStart.textContent = "Camera On";
    btnFlush.disabled = false;
    isRunning = true;
    connectWS();
    requestAnimationFrame(processFrame);
  } catch (err) {
    btnStart.disabled = false;
    btnStart.textContent = "Start Camera";
    toast(`❌ Camera error: ${err.message}`, 4000);
  }
}

// ── Landmark normalisation ────────────────────────────────────────────────────
function normaliseLandmarks(landmarks) {
  const wrist = landmarks[0];
  const centred = landmarks.map((lm) => ({
    x: lm.x - wrist.x,
    y: lm.y - wrist.y,
    z: lm.z - wrist.z,
  }));
  const tip = centred[12];
  const scale = Math.sqrt(tip.x ** 2 + tip.y ** 2 + tip.z ** 2) || 1;
  return centred.flatMap((lm) => [lm.x / scale, lm.y / scale, lm.z / scale]);
}

// ── Draw landmarks on canvas ──────────────────────────────────────────────────
function drawOverlay(result) {
  canvas.width  = video.videoWidth  || canvas.offsetWidth;
  canvas.height = video.videoHeight || canvas.offsetHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!result.landmarks?.length) return;

  const du = new DrawingUtils(ctx);
  for (const handLandmarks of result.landmarks) {
    du.drawConnectors(handLandmarks, HandLandmarker.HAND_CONNECTIONS, {
      color: "rgba(0, 229, 160, 0.55)",
      lineWidth: 2,
    });
    du.drawLandmarks(handLandmarks, {
      color: "rgba(0, 180, 255, 0.9)",
      radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1),
      lineWidth: 1,
    });
  }
}

// ── Main frame loop ───────────────────────────────────────────────────────────
function processFrame() {
  if (!isRunning) return;
  rafId = requestAnimationFrame(processFrame);

  if (!handLandmarker || video.readyState < 2) return;

  const result = handLandmarker.detectForVideo(video, performance.now());
  drawOverlay(result);

  // Build normalised feature vector (use first hand if present)
  if (result.landmarks?.length > 0) {
    const norm = normaliseLandmarks(result.landmarks[0]);
    if (norm.length === NUM_FEATURES) {
      frameBuffer.push(norm);
      if (frameBuffer.length > FRAME_SIZE) frameBuffer.shift();
    }
  } else {
    // No hand visible — inject zero frame so model sees absence
    frameBuffer.push(new Array(NUM_FEATURES).fill(0));
    if (frameBuffer.length > FRAME_SIZE) frameBuffer.shift();
  }

  frameCount++;

  // Send to backend every STEP_SIZE frames when buffer is full
  if (
    frameBuffer.length === FRAME_SIZE &&
    frameCount % STEP_SIZE === 0 &&
    ws?.readyState === 1
  ) {
    ws.send(JSON.stringify({
      action: "predict",
      frames: frameBuffer.map((f) => [...f]),
    }));
  }
}

// ── HUD update ────────────────────────────────────────────────────────────────
function updateHUD(sign, conf) {
  const thresh = parseInt(confThresh.value) / 100;

  if (!sign || conf < thresh) {
    hudSign.textContent = "—";
    hudSign.style.color = "var(--text2)";
    confBar.style.width = `${Math.round(conf * 100)}%`;
    confBar.className   = "conf-bar-fill low";
    confPct.textContent = `${Math.round(conf * 100)}%`;
    return;
  }

  hudSign.textContent = sign;
  hudSign.style.color = "var(--accent)";

  const pct = Math.round(conf * 100);
  confBar.style.width = `${pct}%`;
  confBar.className   = `conf-bar-fill ${pct >= 85 ? "" : pct >= 72 ? "mid" : "low"}`;
  confPct.textContent = `${pct}%`;
}

// ── Sign buffer chips ─────────────────────────────────────────────────────────
function updateSignBuffer(signs) {
  if (!signs.length) {
    signBuf.innerHTML = '<span class="buffer-empty">Start signing…</span>';
    return;
  }
  signBuf.innerHTML = signs
    .map((s, i) =>
      `<span class="sign-chip${i === signs.length - 1 ? " latest" : ""}">${s}</span>`
    )
    .join("");
}

// ── Sentence card ─────────────────────────────────────────────────────────────
function addSentence(sentence, gloss) {
  lastSentence = sentence;
  btnSpeak.disabled = false;

  // Remove placeholder
  const placeholder = sentList.querySelector(".sentence-placeholder");
  if (placeholder) placeholder.remove();

  const now  = new Date().toLocaleTimeString();
  const card = document.createElement("div");
  card.className = "sentence-card";
  card.innerHTML = `
    <div class="sentence-text">${sentence}</div>
    <div class="sentence-meta">${now}  ·  gloss: ${gloss}</div>
  `;
  sentList.prepend(card);
  sentList.scrollTop = 0;

  // Gloss log
  const chip = document.createElement("span");
  chip.textContent = gloss;
  const entry = document.createElement("div");
  entry.appendChild(chip);
  glossLog.prepend(entry);

  // Auto-speak
  speak(sentence);
}

// ── Text-to-speech ────────────────────────────────────────────────────────────
function speak(text) {
  if (!text || !window.speechSynthesis) return;
  speechSynthesis.cancel();
  const utt   = new SpeechSynthesisUtterance(text);
  utt.rate    = 0.95;
  utt.pitch   = 1.0;
  const voices = speechSynthesis.getVoices();
  const nat    = voices.find((v) => /natural|premium|enhanced/i.test(v.name));
  if (nat) utt.voice = nat;
  speechSynthesis.speak(utt);
}

// ── Button handlers ───────────────────────────────────────────────────────────
btnStart.addEventListener("click", async () => {
  if (!handLandmarker) await initMediaPipe();
  await startCamera();
});

btnFlush.addEventListener("click", () => {
  if (ws?.readyState === 1) {
    ws.send(JSON.stringify({ action: "flush" }));
  }
  toast("Sentence flushed");
});

btnClear.addEventListener("click", () => {
  sentList.innerHTML = '<div class="sentence-placeholder">Completed sentences will appear here…</div>';
  glossLog.innerHTML = "";
  updateSignBuffer([]);
  updateHUD(null, 0);
  lastSentence = "";
  btnSpeak.disabled = true;
  toast("Cleared");
});

btnSpeak.addEventListener("click", () => {
  if (lastSentence) speak(lastSentence);
});

// Settings sliders
confThresh.addEventListener("input", () => {
  confThreshV.textContent = `${confThresh.value}%`;
});
pauseThresh.addEventListener("input", () => {
  const val = (parseInt(pauseThresh.value) / 10).toFixed(1);
  pauseThreshV.textContent = `${val}s`;
  // Update backend pause threshold via WS if connected
  if (ws?.readyState === 1) {
    ws.send(JSON.stringify({ action: "set_pause", value: parseFloat(val) }));
  }
});

// ── Boot ──────────────────────────────────────────────────────────────────────
checkBackend();
// Preload voices for speech synthesis
window.speechSynthesis?.getVoices();
speechSynthesis.addEventListener?.("voiceschanged", () => speechSynthesis.getVoices());
