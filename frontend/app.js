/**
 * ASL Translator — Full-screen overlay layout
 * Camera fills the screen. Minimal HUD. Controls auto-hide after 3s.
 */

import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.js";

// ── Constants ─────────────────────────────────────────────────────────────────
const FRAME_SIZE      = 30;
const STEP_SIZE       = 10;
const NUM_FEATURES    = 63;
const CLIENT_ID       = Math.random().toString(36).slice(2);
const CONTROLS_TIMEOUT = 3000; // ms before controls hide

// ── Pre-allocations ───────────────────────────────────────────────────────────
const ringBuffer = new Float32Array(FRAME_SIZE * NUM_FEATURES);
let   ringHead   = 0;
let   ringFilled = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
function getApiBase() { return $('backendUrl')?.value?.trim().replace(/\/$/, '') || 'http://localhost:8000'; }
function getWsBase()  { return getApiBase().replace(/^http/, 'ws'); }

// ── DOM ───────────────────────────────────────────────────────────────────────
const video          = $('video');
const canvas         = $('overlay');
const ctx            = canvas.getContext('2d');
const startScreen    = $('startScreen');
const topBar         = $('topBar');
const topDot         = $('topDot');
const topStatus      = $('topStatus');
const handRing       = $('handRing');
const signWord       = $('signWord');
const sentenceText   = $('sentenceText');
const sentenceChips  = $('sentenceChips');
const controls       = $('controls');
const settingsPanel  = $('settingsPanel');
const historyPanel   = $('historyPanel');
const historyList    = $('historyList');
const modelInfo      = $('modelInfo');
const statusDot      = $('statusDot');
const statusText     = $('statusText');
const btnStart       = $('btnStart');
const btnFlush       = $('btnFlush');
const btnSpeak       = $('btnSpeak');
const btnClear       = $('btnClear');
const btnSettings    = $('btnSettings');
const btnSettingsClose = $('btnSettingsClose');
const btnHistoryClose  = $('btnHistoryClose');
const confThresh     = $('confThresh');
const confThreshVal  = $('confThreshVal');
const pauseThresh    = $('pauseThresh');
const pauseThreshVal = $('pauseThreshVal');
const toastEl        = $('toast');
const celebCanvas    = $('celebrationCanvas');

let celebCtx = null;

// ── State ─────────────────────────────────────────────────────────────────────
let handLandmarker   = null;
let ws               = null;
let frameCount       = 0;
let isRunning        = false;
let lastSentence     = '';
let sentenceTotal    = 0;
let handVisible      = false;
let lastSignText     = '';
let controlsHideTimer = null;
let mpReady          = false;

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer;
function toast(msg, ms = 2600) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), ms);
}

// ── Ripple ────────────────────────────────────────────────────────────────────
function addRipple(btn, e) {
  const rect = btn.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);
  const r    = document.createElement('span');
  r.className = 'ripple';
  r.style.cssText = `width:${size}px;height:${size}px;left:${(e.clientX-rect.left)-size/2}px;top:${(e.clientY-rect.top)-size/2}px`;
  btn.appendChild(r);
  r.addEventListener('animationend', () => r.remove(), { once: true });
}
btnStart.addEventListener('click', e => addRipple(btnStart, e));

// ── Controls auto-hide ────────────────────────────────────────────────────────
function showControls() {
  controls.classList.remove('hidden');
  topBar.classList.remove('hidden');
  clearTimeout(controlsHideTimer);
  controlsHideTimer = setTimeout(() => {
    controls.classList.add('hidden');
    topBar.classList.add('hidden');
  }, CONTROLS_TIMEOUT);
}

function resetHideTimer() {
  if (!isRunning) return;
  showControls();
}

// Tap anywhere on screen to show controls
document.addEventListener('pointerdown', (e) => {
  // Don't reset if tapping a panel or button directly
  if (e.target.closest('.settings-panel, .history-panel, .start-screen')) return;
  resetHideTimer();
}, { passive: true });

// ── Status ────────────────────────────────────────────────────────────────────
function setStatus(state, text) {
  const cls = `sdot${state ? ' '+state : ''}`;
  if (statusDot) statusDot.className = cls;
  if (topDot)    topDot.className    = cls;
  if (statusText) statusText.textContent = text;
  if (topStatus)  topStatus.textContent  = text;
}

// ── Backend check ─────────────────────────────────────────────────────────────
async function checkBackend() {
  setStatus('', 'Checking…');
  try {
    const r = await fetch(`${getApiBase()}/status`, { signal: AbortSignal.timeout(3500) });
    const d = await r.json();
    if (d.model_loaded) {
      setStatus('ok', `${d.num_signs} signs`);
      const info = `✓ ${d.num_signs} signs loaded\n${d.signs.join(', ')}`;
      if (modelInfo) modelInfo.textContent = info;
    } else {
      setStatus('warn', 'No model');
      if (modelInfo) modelInfo.textContent = '⚠ No model\nRun train_model.py first';
      toast('⚠ No trained model — run train_model.py first', 5000);
    }
  } catch {
    setStatus('error', 'Offline');
    if (modelInfo) modelInfo.textContent = '✕ Backend offline\nuvicorn main:app --reload';
    toast('✕ Backend offline — start the server first', 5000);
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS() {
  if (ws && ws.readyState <= 1) return;
  ws = new WebSocket(`${getWsBase()}/ws/${CLIENT_ID}`);
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'prediction') { updateHUD(msg.sign, msg.confidence); updateChips(msg.buffer || []); }
    if (msg.type === 'sentence')   { addSentence(msg.sentence, msg.gloss); updateChips([]); }
  };
  ws.onclose = () => setTimeout(connectWS, 3000);
}

// ── MediaPipe — idle prefetch ─────────────────────────────────────────────────
function prefetchMediaPipe() {
  if (mpReady) return;
  FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  ).then(v => { window.__mpVision = v; }).catch(() => {});
}
if ('requestIdleCallback' in window) requestIdleCallback(prefetchMediaPipe, { timeout: 3000 });
else setTimeout(prefetchMediaPipe, 1200);

async function initMediaPipe() {
  if (mpReady) return;
  toast('Loading hand detection…');
  const vision = window.__mpVision || await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO', numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence:  0.50,
    minTrackingConfidence:      0.50,
  });
  mpReady = true;
  toast('Ready — show your hands ✋');
}

// ── Camera ────────────────────────────────────────────────────────────────────
async function startCamera() {
  btnStart.disabled = true;
  btnStart.querySelector('.btn-label').textContent = 'Starting…';

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    // Fade out start screen
    startScreen.classList.add('gone');

    btnFlush.disabled = false;
    isRunning = true;
    connectWS();
    requestAnimationFrame(processFrame);

    // Show controls briefly then auto-hide
    showControls();
  } catch (err) {
    btnStart.disabled = false;
    btnStart.querySelector('.btn-label').textContent = 'Start Camera';
    toast(`✕ Camera blocked: ${err.message}`, 5000);
  }
}

// ── Normalise into ring buffer ────────────────────────────────────────────────
function normaliseIntoRing(lms) {
  const w    = lms[0];
  const base = ringHead * NUM_FEATURES;
  for (let i = 0; i < 21; i++) {
    const lm = lms[i];
    ringBuffer[base+i*3]   = lm.x - w.x;
    ringBuffer[base+i*3+1] = lm.y - w.y;
    ringBuffer[base+i*3+2] = lm.z - w.z;
  }
  const tx = ringBuffer[base+36], ty = ringBuffer[base+37], tz = ringBuffer[base+38];
  const s  = Math.sqrt(tx*tx+ty*ty+tz*tz) || 1;
  for (let j = 0; j < NUM_FEATURES; j++) ringBuffer[base+j] /= s;
}

function writeZeroFrame() { ringBuffer.fill(0, ringHead*NUM_FEATURES, ringHead*NUM_FEATURES+NUM_FEATURES); }

function advanceRing() {
  ringHead = (ringHead+1) % FRAME_SIZE;
  if (ringFilled < FRAME_SIZE) ringFilled++;
}

function ringToArray() {
  const out = [];
  for (let i = 0; i < FRAME_SIZE; i++) {
    const idx = ((ringHead-FRAME_SIZE+i+FRAME_SIZE) % FRAME_SIZE) * NUM_FEATURES;
    const row = [];
    for (let j = 0; j < NUM_FEATURES; j++) row.push(ringBuffer[idx+j]);
    out.push(row);
  }
  return out;
}

// ── Draw landmarks ────────────────────────────────────────────────────────────
function drawOverlay(result) {
  const vw = video.videoWidth  || canvas.offsetWidth;
  const vh = video.videoHeight || canvas.offsetHeight;
  if (canvas.width !== vw)  canvas.width  = vw;
  if (canvas.height !== vh) canvas.height = vh;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const hasHand = result.landmarks?.length > 0;
  if (hasHand !== handVisible) {
    handVisible = hasHand;
    handRing.classList.toggle('active', hasHand);
  }
  if (!hasHand) return;

  const du = new DrawingUtils(ctx);
  for (const hl of result.landmarks) {
    du.drawConnectors(hl, HandLandmarker.HAND_CONNECTIONS, { color: 'rgba(41,196,154,.3)', lineWidth: 1.5 });
    du.drawLandmarks(hl, {
      color: 'rgba(41,196,154,.75)',
      radius: d => DrawingUtils.lerp(d.from?.z ?? 0, -0.15, 0.1, 5, 1),
      lineWidth: 1,
    });
  }
}

// ── Frame loop ────────────────────────────────────────────────────────────────
function processFrame() {
  if (!isRunning) return;
  requestAnimationFrame(processFrame);
  if (!handLandmarker || video.readyState < 2) return;

  const result = handLandmarker.detectForVideo(video, performance.now());
  drawOverlay(result);

  if (result.landmarks?.length > 0) normaliseIntoRing(result.landmarks[0]);
  else writeZeroFrame();
  advanceRing();

  frameCount++;
  if (ringFilled === FRAME_SIZE && frameCount % STEP_SIZE === 0 && ws?.readyState === 1) {
    ws.send(JSON.stringify({ action: 'predict', frames: ringToArray() }));
  }
}

// ── HUD ───────────────────────────────────────────────────────────────────────
let lastChipSigns = [];

function updateHUD(sign, conf) {
  const thresh  = parseInt(confThresh.value) / 100;
  const valid   = !!(sign && conf >= thresh);
  const newSign = valid && sign !== lastSignText;

  if (newSign) lastSignText = sign;
  if (!valid)  lastSignText = '';

  signWord.textContent = valid ? sign : '—';
  signWord.className   = `sign-word${valid ? '' : ' dim'}${newSign ? ' pop' : ''}`;
  if (newSign) setTimeout(() => signWord.classList.remove('pop'), 280);
}

function updateChips(signs) {
  if (signs.length === lastChipSigns.length && signs.every((s,i) => s === lastChipSigns[i])) return;
  lastChipSigns = [...signs];

  if (!signs.length) { sentenceChips.innerHTML = ''; return; }
  sentenceChips.innerHTML = signs.map((s, i) =>
    `<span class="s-chip${i === signs.length-1 ? ' new' : ''}">${s}</span>`
  ).join('');
}

// ── Sentence ──────────────────────────────────────────────────────────────────
function addSentence(sentence, gloss) {
  lastSentence = sentence;
  sentenceTotal++;
  const isFirst     = sentenceTotal === 1;
  const isMilestone = sentenceTotal % 5 === 0;

  btnSpeak.disabled = false;

  // Update bottom bar
  sentenceText.textContent = sentence;
  sentenceText.classList.remove('placeholder');

  // Add to history
  historyList.querySelector('.history-empty')?.remove();
  const card = document.createElement('div');
  card.className = 'h-card';
  card.innerHTML = `<div class="h-card-text">${sentence}</div>
    <div class="h-card-meta">${new Date().toLocaleTimeString()} &nbsp;·&nbsp; <span class="h-card-gloss">${gloss}</span></div>`;
  historyList.prepend(card);

  if (isFirst || isMilestone) {
    celebrate();
    toast(isFirst ? 'First translation ✓' : `${sentenceTotal} translations 🎉`, 2000);
  }

  speak(sentence);
  showControls();
}

// ── TTS ───────────────────────────────────────────────────────────────────────
function speak(text) {
  if (!text || !window.speechSynthesis) return;
  speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate = 0.95; utt.pitch = 1.0;
  const nat = speechSynthesis.getVoices().find(v => /natural|premium|enhanced/i.test(v.name));
  if (nat) utt.voice = nat;
  speechSynthesis.speak(utt);
}

// ── Celebration ───────────────────────────────────────────────────────────────
function celebrate() {
  if (!celebCtx) {
    celebCtx = celebCanvas.getContext('2d');
  }
  if (celebCanvas.width !== window.innerWidth) celebCanvas.width = window.innerWidth;
  if (celebCanvas.height !== window.innerHeight) celebCanvas.height = window.innerHeight;
  celebCanvas.style.display = 'block';

  const colors = ['#29c49a','#4dd9b4','#1a9e7c','#a8f0dc','#e8e8ec'];
  const pieces = Array.from({ length: 60 }, () => ({
    x: Math.random() * celebCanvas.width, y: -10,
    vx: (Math.random()-.5)*3.5, vy: Math.random()*4+2,
    rot: Math.random()*360, vr: (Math.random()-.5)*6,
    w: Math.random()*6+3, h: Math.random()*3+2,
    c: colors[Math.floor(Math.random()*colors.length)], a: 1,
  }));

  let frame = 0;
  function draw() {
    celebCtx.clearRect(0, 0, celebCanvas.width, celebCanvas.height);
    let alive = false;
    for (const p of pieces) {
      p.x+=p.vx; p.y+=p.vy; p.vy+=.12; p.rot+=p.vr;
      if (frame > 40) p.a -= .025;
      if (p.a > 0) {
        alive = true;
        celebCtx.save();
        celebCtx.globalAlpha = p.a;
        celebCtx.translate(p.x,p.y); celebCtx.rotate(p.rot*Math.PI/180);
        celebCtx.fillStyle = p.c;
        celebCtx.fillRect(-p.w/2,-p.h/2,p.w,p.h);
        celebCtx.restore();
      }
    }
    frame++;
    if (alive) requestAnimationFrame(draw);
    else celebCanvas.style.display = 'none';
  }
  requestAnimationFrame(draw);
}

// ── Clear ─────────────────────────────────────────────────────────────────────
function clearAll() {
  sentenceText.textContent = 'Completed sentences appear here';
  sentenceText.classList.add('placeholder');
  sentenceChips.innerHTML = '';
  updateHUD(null, 0);
  lastSentence = ''; sentenceTotal = 0; lastSignText = ''; lastChipSigns = [];
  historyList.innerHTML = '<div class="history-empty">No translations yet</div>';
  btnSpeak.disabled = true;
  toast('Cleared');
}

// ── Flush ─────────────────────────────────────────────────────────────────────
function flush() {
  ws?.readyState === 1 && ws.send(JSON.stringify({ action: 'flush' }));
  toast('Sentence flushed');
}

// ── Settings panel ────────────────────────────────────────────────────────────
btnSettings.addEventListener('click', () => {
  settingsPanel.classList.add('open');
  clearTimeout(controlsHideTimer);
});
btnSettingsClose.addEventListener('click', () => {
  settingsPanel.classList.remove('open');
  showControls();
});
settingsPanel.addEventListener('click', (e) => {
  if (e.target === settingsPanel) settingsPanel.classList.remove('open');
});

// ── History panel — tap sentence bar to open ──────────────────────────────────
sentenceText.addEventListener('click', () => {
  if (sentenceTotal > 0) historyPanel.classList.add('open');
});
btnHistoryClose.addEventListener('click', () => historyPanel.classList.remove('open'));
historyPanel.addEventListener('click', (e) => {
  if (e.target === historyPanel) historyPanel.classList.remove('open');
});

// ── Sliders ───────────────────────────────────────────────────────────────────
confThresh.addEventListener('input', () => {
  confThreshVal.textContent = `${confThresh.value}%`;
});
pauseThresh.addEventListener('input', () => {
  const val = (parseInt(pauseThresh.value)/10).toFixed(1);
  pauseThreshVal.textContent = `${val}s`;
  ws?.readyState === 1 && ws.send(JSON.stringify({ action: 'set_pause', value: parseFloat(val) }));
});

// ── Buttons ───────────────────────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  if (!mpReady) await initMediaPipe();
  await startCamera();
});
btnFlush.addEventListener('click', flush);
btnClear.addEventListener('click', clearAll);
btnSpeak.addEventListener('click', () => { if (lastSentence) speak(lastSentence); });

// ── Boot ──────────────────────────────────────────────────────────────────────
sentenceText.classList.add('placeholder');
checkBackend();
window.speechSynthesis?.getVoices();
speechSynthesis.addEventListener?.('voiceschanged', () => speechSynthesis.getVoices());
console.log('%c✋ ASL Translator', 'font-size:18px;font-weight:bold;color:#29c49a');
