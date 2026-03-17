/**
 * ASL Translator — app.js
 * Drives both desktop and mobile layouts from a single state.
 * Desktop elements have "d" prefix, mobile have "m" prefix.
 */

import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.js";

// ── Constants ─────────────────────────────────────────────────────────────────
const FRAME_SIZE   = 30;
const STEP_SIZE    = 10;
const NUM_FEATURES = 63;
const CLIENT_ID    = Math.random().toString(36).slice(2);

// ── Helpers ───────────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const isDesktop = () => window.innerWidth >= 1024;

function getApiBase() {
  const url = $('mBackendUrl')?.value?.trim().replace(/\/$/, '');
  return url || 'http://localhost:8000';
}
function getWsBase() { return getApiBase().replace(/^http/, 'ws'); }

// ── DOM — desktop ─────────────────────────────────────────────────────────────
const dVideo       = $('d-video');
const dCanvas      = $('d-overlay');
const dCtx         = dCanvas?.getContext('2d');
const dPlaceholder = $('dPlaceholder');
const dSignWord    = $('dSignWord');
const dConfBar     = $('dConfBar');
const dConfPct     = $('dConfPct');
const dChipRow     = $('dChipRow');
const dSentList    = $('dSentenceList');
const dDot         = $('dDot');
const dStatusText  = $('dStatusText');
const dSignCount   = $('dSignCount');
const dModelInfo   = $('dModelInfo');
const dBtnStart    = $('dBtnStart');
const dBtnFlush    = $('dBtnFlush');
const dBtnSpeak    = $('dBtnSpeak');
const dBtnClear    = $('dBtnClear');
const dConfThresh  = $('dConfThresh');
const dConfThreshV = $('dConfThreshVal');
const dPauseThresh = $('dPauseThresh');
const dPauseThreshV= $('dPauseThreshVal');

// ── DOM — mobile ──────────────────────────────────────────────────────────────
const mVideo       = $('m-video');
const mCanvas      = $('m-overlay');
const mCtx         = mCanvas?.getContext('2d');
const mPlaceholder = $('mPlaceholder');
const mSignWord    = $('mSignWord');
const mConfBar     = $('mConfBar');
const mConfPct     = $('mConfPct');
const mChipRow     = $('mChipRow');
const mSentList    = $('mSentenceList');
const mDot         = $('mDot');
const mStatusText  = $('mStatusText');
const mModelInfo   = $('mModelInfo');
const mBtnStart    = $('mBtnStart');
const mBtnFlush    = $('mBtnFlush');
const mBtnSpeak    = $('mBtnSpeak');
const mBtnClear    = $('mBtnClear');
const mBadge       = $('mBadge');
const mConfThresh  = $('mConfThresh');
const mConfThreshV = $('mConfThreshVal');
const mPauseThresh = $('mPauseThresh');
const mPauseThreshV= $('mPauseThreshVal');
const toastEl      = $('toast');

// ── State ─────────────────────────────────────────────────────────────────────
let handLandmarker = null;
let ws             = null;
let frameBuffer    = [];
let frameCount     = 0;
let isRunning      = false;
let lastSentence   = '';
let unreadCount    = 0;
let activeVideo    = null;   // whichever video element is active
let activeCanvas   = null;
let activeCtx      = null;

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer;
function toast(msg, ms = 2800) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), ms);
}

// ── Set status on both layouts ────────────────────────────────────────────────
function setStatus(state, text, signCountText) {
  [dDot, mDot].forEach(el => { if (el) el.className = state ? `${el.className.split(' ')[0]} ${state}` : el.className.split(' ')[0]; });
  if (dDot) dDot.className = `d-dot${state ? ' ' + state : ''}`;
  if (mDot) mDot.className = `m-dot${state ? ' ' + state : ''}`;
  if (dStatusText) dStatusText.textContent = text;
  if (mStatusText) mStatusText.textContent = text;
  if (dSignCount && signCountText) dSignCount.textContent = signCountText;
}

// ── Backend check ─────────────────────────────────────────────────────────────
async function checkBackend() {
  try {
    const r = await fetch(`${getApiBase()}/status`, { signal: AbortSignal.timeout(3500) });
    const d = await r.json();
    if (d.model_loaded) {
      setStatus('ok', 'Ready', `${d.num_signs} signs`);
      const info = `✓ ${d.num_signs} signs loaded\n${d.signs.join(', ')}`;
      if (dModelInfo) dModelInfo.textContent = info;
      if (mModelInfo) mModelInfo.textContent = info;
    } else {
      setStatus('warn', 'No model', '0 signs');
      const info = '⚠ No model found\nRun train_model.py first';
      if (dModelInfo) dModelInfo.textContent = info;
      if (mModelInfo) mModelInfo.textContent = info;
      toast('⚠ No trained model — run train_model.py first', 5000);
    }
  } catch {
    setStatus('error', 'Offline', '—');
    const info = '✕ Backend offline\nuvicorn main:app --reload';
    if (dModelInfo) dModelInfo.textContent = info;
    if (mModelInfo) mModelInfo.textContent = info;
    toast('✕ Backend offline — start the server first', 5000);
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS() {
  if (ws && ws.readyState <= 1) return;
  ws = new WebSocket(`${getWsBase()}/ws/${CLIENT_ID}`);
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'prediction') {
      updateHUD(msg.sign, msg.confidence);
      updateChips(msg.buffer || []);
    }
    if (msg.type === 'sentence') {
      addCard(msg.sentence, msg.gloss);
      updateChips([]);
    }
  };
  ws.onclose = () => setTimeout(connectWS, 3000);
}

// ── MediaPipe ─────────────────────────────────────────────────────────────────
async function initMediaPipe() {
  toast('Loading hand detection…');
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence:  0.50,
    minTrackingConfidence:      0.50,
  });
  toast('Hand detection ready ✓');
}

// ── Start camera ──────────────────────────────────────────────────────────────
async function startCamera() {
  // Determine active video/canvas based on current layout
  activeVideo  = isDesktop() ? dVideo  : mVideo;
  activeCanvas = isDesktop() ? dCanvas : mCanvas;
  activeCtx    = isDesktop() ? dCtx    : mCtx;

  // Disable both start buttons while starting
  [dBtnStart, mBtnStart].forEach(b => { if (b) { b.disabled = true; b.textContent = 'Starting…'; }});

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });

    // Attach stream to BOTH video elements so switching layout works
    if (dVideo)  { dVideo.srcObject  = stream; await dVideo.play().catch(()=>{}); }
    if (mVideo)  { mVideo.srcObject  = stream; await mVideo.play().catch(()=>{}); }

    if (dPlaceholder) dPlaceholder.classList.add('gone');
    if (mPlaceholder) mPlaceholder.classList.add('gone');

    [dBtnStart, mBtnStart].forEach(b => { if (b) { b.textContent = 'Camera On'; b.disabled = false; }});
    [dBtnFlush, mBtnFlush].forEach(b => { if (b) b.disabled = false; });

    isRunning = true;
    connectWS();
    requestAnimationFrame(processFrame);
  } catch (err) {
    [dBtnStart, mBtnStart].forEach(b => { if (b) { b.disabled = false; b.textContent = 'Start Camera'; }});
    toast(`✕ Camera error: ${err.message}`, 5000);
  }
}

// ── Normalise ─────────────────────────────────────────────────────────────────
function normalise(lms) {
  const w = lms[0];
  const c = lms.map(l => ({ x: l.x - w.x, y: l.y - w.y, z: l.z - w.z }));
  const t = c[12];
  const s = Math.sqrt(t.x**2 + t.y**2 + t.z**2) || 1;
  return c.flatMap(l => [l.x/s, l.y/s, l.z/s]);
}

// ── Draw landmarks ────────────────────────────────────────────────────────────
function drawOn(video, canvas, ctx, result) {
  if (!canvas || !ctx) return;
  canvas.width  = video.videoWidth  || canvas.offsetWidth;
  canvas.height = video.videoHeight || canvas.offsetHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!result.landmarks?.length) return;
  const du = new DrawingUtils(ctx);
  for (const hl of result.landmarks) {
    du.drawConnectors(hl, HandLandmarker.HAND_CONNECTIONS, { color: 'rgba(45,212,170,.4)', lineWidth: 1.5 });
    du.drawLandmarks(hl, {
      color: 'rgba(45,212,170,.85)',
      radius: d => DrawingUtils.lerp(d.from?.z ?? 0, -0.15, 0.1, 4, 1),
      lineWidth: 1,
    });
  }
}

// ── Frame loop ────────────────────────────────────────────────────────────────
function processFrame() {
  if (!isRunning) return;
  requestAnimationFrame(processFrame);

  // Always use whichever video is currently visible
  const vid = isDesktop() ? dVideo : mVideo;
  const can = isDesktop() ? dCanvas : mCanvas;
  const cx  = isDesktop() ? dCtx   : mCtx;

  if (!handLandmarker || !vid || vid.readyState < 2) return;

  const result = handLandmarker.detectForVideo(vid, performance.now());
  drawOn(vid, can, cx, result);

  const norm = result.landmarks?.length > 0
    ? normalise(result.landmarks[0])
    : new Array(NUM_FEATURES).fill(0);

  if (norm.length === NUM_FEATURES) {
    frameBuffer.push(norm);
    if (frameBuffer.length > FRAME_SIZE) frameBuffer.shift();
  }

  frameCount++;
  if (frameBuffer.length === FRAME_SIZE && frameCount % STEP_SIZE === 0 && ws?.readyState === 1) {
    ws.send(JSON.stringify({ action: 'predict', frames: frameBuffer.map(f => [...f]) }));
  }
}

// ── HUD update ────────────────────────────────────────────────────────────────
function getThreshold() {
  const src = isDesktop() ? dConfThresh : mConfThresh;
  return parseInt(src?.value ?? 72) / 100;
}

function updateHUD(sign, conf) {
  const thresh = getThreshold();
  const pct    = Math.round(conf * 100);
  const cls    = pct >= 85 ? '' : pct >= 72 ? ' mid' : ' low';
  const valid  = sign && conf >= thresh;

  // desktop
  if (dSignWord) {
    dSignWord.textContent = valid ? sign : '—';
    dSignWord.className   = `d-sign-word${valid ? '' : ' dim'}`;
  }
  if (dConfBar) { dConfBar.style.width = `${pct}%`; dConfBar.className = `d-conf-bar${cls}`; }
  if (dConfPct) dConfPct.textContent = `${pct}%`;

  // mobile
  if (mSignWord) {
    mSignWord.textContent = valid ? sign : '—';
    mSignWord.className   = `m-sign-word${valid ? '' : ' dim'}`;
  }
  if (mConfBar) { mConfBar.style.width = `${pct}%`; mConfBar.className = `m-conf-bar${cls}`; }
  if (mConfPct) mConfPct.textContent = `${pct}%`;
}

// ── Chips ─────────────────────────────────────────────────────────────────────
function updateChips(signs) {
  const empty = '<span class="d-empty">Signing appears here…</span>';
  const mempty= '<span class="m-empty">Start signing…</span>';

  if (!signs.length) {
    if (dChipRow) dChipRow.innerHTML = empty;
    if (mChipRow) mChipRow.innerHTML = mempty;
    return;
  }
  const html = (prefix) => signs.map((s, i) =>
    `<span class="${prefix}-chip${i === signs.length-1 ? ' new' : ''}">${s}</span>`
  ).join('');

  if (dChipRow) dChipRow.innerHTML = html('d');
  if (mChipRow) mChipRow.innerHTML = html('m');
}

// ── Sentence card ─────────────────────────────────────────────────────────────
function addCard(sentence, gloss) {
  lastSentence = sentence;
  [dBtnSpeak, mBtnSpeak].forEach(b => { if (b) b.disabled = false; });

  const time = new Date().toLocaleTimeString();

  // Desktop card
  if (dSentList) {
    dSentList.querySelector('.d-no-sentences')?.remove();
    const c = document.createElement('div');
    c.className = 'd-card';
    c.innerHTML = `<div class="d-card-text">${sentence}</div>
      <div class="d-card-meta">${time} &nbsp;·&nbsp; <span class="d-card-gloss">${gloss}</span></div>`;
    dSentList.prepend(c);
  }

  // Mobile card
  if (mSentList) {
    mSentList.querySelector('.m-no-sentences')?.remove();
    const c = document.createElement('div');
    c.className = 'm-card';
    c.innerHTML = `<div class="m-card-text">${sentence}</div>
      <div class="m-card-meta">${time} &nbsp;·&nbsp; <span class="m-card-gloss">${gloss}</span></div>`;
    mSentList.prepend(c);
  }

  // Badge on mobile sentences tab if not active
  const sentTab = document.querySelector('[data-tab="sentences"]');
  if (sentTab && !sentTab.classList.contains('active')) {
    unreadCount++;
    if (mBadge) { mBadge.textContent = unreadCount; mBadge.classList.add('show'); }
  }

  speak(sentence);

  // Auto-switch to sentences tab on mobile
  if (!isDesktop() && sentTab) sentTab.click();
}

// ── TTS ───────────────────────────────────────────────────────────────────────
function speak(text) {
  if (!text || !window.speechSynthesis) return;
  speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate = 0.95; utt.pitch = 1.0;
  const voices = speechSynthesis.getVoices();
  const nat    = voices.find(v => /natural|premium|enhanced/i.test(v.name));
  if (nat) utt.voice = nat;
  speechSynthesis.speak(utt);
}

// ── Clear all ─────────────────────────────────────────────────────────────────
function clearAll() {
  if (dSentList) dSentList.innerHTML = '<div class="d-no-sentences">Completed sentences appear here</div>';
  if (mSentList) mSentList.innerHTML = '<div class="m-no-sentences">Completed sentences appear here</div>';
  updateChips([]); updateHUD(null, 0);
  lastSentence = ''; unreadCount = 0;
  if (mBadge) mBadge.classList.remove('show');
  [dBtnSpeak, mBtnSpeak].forEach(b => { if (b) b.disabled = true; });
  toast('Cleared');
}

// ── Flush ─────────────────────────────────────────────────────────────────────
function flush() {
  ws?.readyState === 1 && ws.send(JSON.stringify({ action: 'flush' }));
  toast('Sentence flushed');
}

// ── Mobile tab switching ──────────────────────────────────────────────────────
document.querySelectorAll('.m-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.m-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.m-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`m-panel-${tab.dataset.tab}`)?.classList.add('active');
    if (tab.dataset.tab === 'sentences') {
      unreadCount = 0;
      if (mBadge) mBadge.classList.remove('show');
    }
  });
});

// ── Slider sync ───────────────────────────────────────────────────────────────
// Sync desktop ↔ mobile sliders so they always match
function syncSlider(srcEl, destEl, display, formatter) {
  if (!srcEl) return;
  srcEl.addEventListener('input', () => {
    const val = srcEl.value;
    if (destEl) destEl.value = val;
    if (display) display.textContent = formatter(val);
    const display2 = srcEl === dConfThresh ? mConfThreshV : mPauseThreshV;
    if (display2 && display2 !== display) display2.textContent = formatter(val);
  });
}

syncSlider(dConfThresh,  mConfThresh,  dConfThreshV,  v => `${v}%`);
syncSlider(mConfThresh,  dConfThresh,  mConfThreshV,  v => `${v}%`);
syncSlider(dPauseThresh, mPauseThresh, dPauseThreshV, v => `${(parseInt(v)/10).toFixed(1)}s`);
syncSlider(mPauseThresh, dPauseThresh, mPauseThreshV, v => {
  const val = `${(parseInt(v)/10).toFixed(1)}s`;
  ws?.readyState === 1 && ws.send(JSON.stringify({ action: 'set_pause', value: parseFloat(val) }));
  return val;
});

// ── Button wiring ─────────────────────────────────────────────────────────────
async function handleStart() {
  if (!handLandmarker) await initMediaPipe();
  await startCamera();
}

[dBtnStart, mBtnStart].forEach(b => b?.addEventListener('click', handleStart));
[dBtnFlush, mBtnFlush].forEach(b => b?.addEventListener('click', flush));
[dBtnClear, mBtnClear].forEach(b => b?.addEventListener('click', clearAll));
[dBtnSpeak, mBtnSpeak].forEach(b => b?.addEventListener('click', () => { if (lastSentence) speak(lastSentence); }));

// ── Boot ──────────────────────────────────────────────────────────────────────
checkBackend();
window.speechSynthesis?.getVoices();
speechSynthesis.addEventListener?.('voiceschanged', () => speechSynthesis.getVoices());
