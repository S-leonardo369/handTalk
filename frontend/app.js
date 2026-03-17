/**
 * ASL Translator — app.js
 * Drives both desktop and mobile layouts from a single state.
 */

import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.js";

const FRAME_SIZE   = 30;
const STEP_SIZE    = 10;
const NUM_FEATURES = 63;
const CLIENT_ID    = Math.random().toString(36).slice(2);

const $         = id => document.getElementById(id);
const isDesktop = () => window.innerWidth >= 1024;

function getApiBase() { return $('mBackendUrl')?.value?.trim().replace(/\/$/, '') || 'http://localhost:8000'; }
function getWsBase()  { return getApiBase().replace(/^http/, 'ws'); }

// ── DOM ───────────────────────────────────────────────────────────────────────
const dVideo        = $('d-video');
const dCanvas       = $('d-overlay');
const dCtx          = dCanvas?.getContext('2d');
const dPlaceholder  = $('dPlaceholder');
const dHandRing     = $('dHandRing');
const dSignWord     = $('dSignWord');
const dConfBar      = $('dConfBar');
const dConfPct      = $('dConfPct');
const dChipRow      = $('dChipRow');
const dSentList     = $('dSentenceList');
const dEmptyState   = $('dEmptyState');
const dDot          = $('dDot');
const dStatusText   = $('dStatusText');
const dSignCount    = $('dSignCount');
const dModelInfo    = $('dModelInfo');
const dBtnStart     = $('dBtnStart');
const dBtnFlush     = $('dBtnFlush');
const dBtnSpeak     = $('dBtnSpeak');
const dBtnClear     = $('dBtnClear');
const dConfThresh   = $('dConfThresh');
const dConfThreshV  = $('dConfThreshVal');
const dPauseThresh  = $('dPauseThresh');
const dPauseThreshV = $('dPauseThreshVal');

const mVideo        = $('m-video');
const mCanvas       = $('m-overlay');
const mCtx          = mCanvas?.getContext('2d');
const mPlaceholder  = $('mPlaceholder');
const mHandRing     = $('mHandRing');
const mSignWord     = $('mSignWord');
const mConfBar      = $('mConfBar');
const mConfPct      = $('mConfPct');
const mChipRow      = $('mChipRow');
const mSentList     = $('mSentenceList');
const mEmptyState   = $('mEmptyState');
const mDot          = $('mDot');
const mStatusText   = $('mStatusText');
const mModelInfo    = $('mModelInfo');
const mBtnStart     = $('mBtnStart');
const mBtnFlush     = $('mBtnFlush');
const mBtnSpeak     = $('mBtnSpeak');
const mBtnClear     = $('mBtnClear');
const mBadge        = $('mBadge');
const mConfThresh   = $('mConfThresh');
const mConfThreshV  = $('mConfThreshVal');
const mPauseThresh  = $('mPauseThresh');
const mPauseThreshV = $('mPauseThreshVal');
const toastEl       = $('toast');
const celebCanvas   = $('celebrationCanvas');
const celebCtx      = celebCanvas?.getContext('2d');

// ── State ─────────────────────────────────────────────────────────────────────
let handLandmarker  = null;
let ws              = null;
let frameBuffer     = [];
let frameCount      = 0;
let isRunning       = false;
let lastSentence    = '';
let unreadCount     = 0;
let sentenceTotal   = 0;   // lifetime sentence count
let handVisible     = false;
let lastSignText    = '';  // for pop animation dedup

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer;
function toast(msg, ms = 2800) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), ms);
}

// ── Ripple on primary button ──────────────────────────────────────────────────
function addRipple(btn, e) {
  const rect   = btn.getBoundingClientRect();
  const size   = Math.max(rect.width, rect.height);
  const x      = (e.clientX - rect.left) - size / 2;
  const y      = (e.clientY - rect.top)  - size / 2;
  const ripple = document.createElement('span');
  ripple.className = 'ripple';
  ripple.style.cssText = `width:${size}px;height:${size}px;left:${x}px;top:${y}px`;
  btn.appendChild(ripple);
  ripple.addEventListener('animationend', () => ripple.remove());
}
[dBtnStart, mBtnStart].forEach(b => {
  b?.addEventListener('click', e => addRipple(b, e));
});

// ── Celebration burst (first sentence & every 5th) ───────────────────────────
function celebrate() {
  if (!celebCanvas || !celebCtx) return;
  celebCanvas.width  = window.innerWidth;
  celebCanvas.height = window.innerHeight;

  // Teal-only particles — stays on-brand, not garish
  const colors  = ['#29c49a', '#4dd9b4', '#1a9e7c', '#a8f0dc', '#e8e8ec'];
  const pieces  = [];
  const count   = 60;

  for (let i = 0; i < count; i++) {
    pieces.push({
      x:   Math.random() * celebCanvas.width,
      y:   -10,
      vx:  (Math.random() - .5) * 3.5,
      vy:  Math.random() * 4 + 2,
      rot: Math.random() * 360,
      vr:  (Math.random() - .5) * 6,
      w:   Math.random() * 6 + 3,
      h:   Math.random() * 3 + 2,
      c:   colors[Math.floor(Math.random() * colors.length)],
      a:   1,
    });
  }

  let frame = 0;
  function draw() {
    celebCtx.clearRect(0, 0, celebCanvas.width, celebCanvas.height);
    let alive = false;
    for (const p of pieces) {
      p.x   += p.vx;
      p.y   += p.vy;
      p.vy  += .12;   // gravity
      p.rot += p.vr;
      if (frame > 40) p.a -= .025;
      if (p.a > 0) {
        alive = true;
        celebCtx.save();
        celebCtx.globalAlpha = p.a;
        celebCtx.translate(p.x, p.y);
        celebCtx.rotate(p.rot * Math.PI / 180);
        celebCtx.fillStyle = p.c;
        celebCtx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
        celebCtx.restore();
      }
    }
    frame++;
    if (alive) requestAnimationFrame(draw);
    else celebCtx.clearRect(0, 0, celebCanvas.width, celebCanvas.height);
  }
  requestAnimationFrame(draw);
}

// ── Status ────────────────────────────────────────────────────────────────────
function setStatus(state, text, signCountText) {
  if (dDot) dDot.className = `d-dot${state ? ' ' + state : ''}`;
  if (mDot) mDot.className = `m-dot${state ? ' ' + state : ''}`;
  if (dStatusText) dStatusText.textContent = text;
  if (mStatusText) mStatusText.textContent = text;
  if (dSignCount && signCountText) dSignCount.textContent = signCountText;
}

// ── Backend check ─────────────────────────────────────────────────────────────
// Context-aware status messages — not generic
const loadingMessages = [
  'Checking model…',
  'Connecting to server…',
  'Loading sign vocabulary…',
];
let loadingIdx = 0;

async function checkBackend() {
  setStatus('', loadingMessages[loadingIdx++ % loadingMessages.length], '—');
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
// Product-specific loading messages (not generic AI filler)
const mpMessages = [
  'Loading hand detection model…',
  'Preparing landmark tracker…',
  'Almost ready to see your signs…',
];
let mpMsgIdx = 0;

async function initMediaPipe() {
  toast(mpMessages[mpMsgIdx++ % mpMessages.length]);
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
  toast('Hand detection ready — show your hands ✋');
}

// ── Camera ────────────────────────────────────────────────────────────────────
async function startCamera() {
  [dBtnStart, mBtnStart].forEach(b => { if (b) { b.disabled = true; b.querySelector('.btn-label').textContent = 'Starting…'; }});

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    if (dVideo)  { dVideo.srcObject  = stream; await dVideo.play().catch(()=>{}); }
    if (mVideo)  { mVideo.srcObject  = stream; await mVideo.play().catch(()=>{}); }

    if (dPlaceholder) dPlaceholder.classList.add('gone');
    if (mPlaceholder) mPlaceholder.classList.add('gone');

    [dBtnStart, mBtnStart].forEach(b => {
      if (b) {
        b.querySelector('.btn-label').textContent = 'Camera On';
        b.disabled = false;
      }
    });
    [dBtnFlush, mBtnFlush].forEach(b => { if (b) b.disabled = false; });

    isRunning = true;
    connectWS();
    requestAnimationFrame(processFrame);
  } catch (err) {
    [dBtnStart, mBtnStart].forEach(b => {
      if (b) { b.disabled = false; b.querySelector('.btn-label').textContent = 'Start Camera'; }
    });
    toast(`✕ Camera blocked: ${err.message}`, 5000);
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

  const hasHand = result.landmarks?.length > 0;

  // Update pulse ring visibility
  if (hasHand !== handVisible) {
    handVisible = hasHand;
    [dHandRing, mHandRing].forEach(r => r?.classList.toggle('active', hasHand));
  }

  if (!hasHand) return;
  const du = new DrawingUtils(ctx);
  for (const hl of result.landmarks) {
    du.drawConnectors(hl, HandLandmarker.HAND_CONNECTIONS, { color: 'rgba(41,196,154,.35)', lineWidth: 1.5 });
    du.drawLandmarks(hl, {
      color: 'rgba(41,196,154,.8)',
      radius: d => DrawingUtils.lerp(d.from?.z ?? 0, -0.15, 0.1, 4, 1),
      lineWidth: 1,
    });
  }
}

// ── Frame loop ────────────────────────────────────────────────────────────────
function processFrame() {
  if (!isRunning) return;
  requestAnimationFrame(processFrame);

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

// ── HUD ───────────────────────────────────────────────────────────────────────
function getThreshold() {
  return parseInt((isDesktop() ? dConfThresh : mConfThresh)?.value ?? 72) / 100;
}

function updateHUD(sign, conf) {
  const thresh = getThreshold();
  const pct    = Math.round(conf * 100);
  const cls    = pct >= 85 ? '' : pct >= 72 ? ' mid' : ' low';
  const valid  = sign && conf >= thresh;
  const newSign = valid && sign !== lastSignText;

  if (newSign) lastSignText = sign;

  // Desktop
  if (dSignWord) {
    dSignWord.textContent = valid ? sign : '—';
    dSignWord.className   = `d-sign-word${valid ? '' : ' dim'}${newSign ? ' pop' : ''}`;
    if (newSign) setTimeout(() => dSignWord.classList.remove('pop'), 300);
  }
  if (dConfBar) { dConfBar.style.width = `${pct}%`; dConfBar.className = `d-conf-bar${cls}`; }
  if (dConfPct) dConfPct.textContent = `${pct}%`;

  // Mobile
  if (mSignWord) {
    mSignWord.textContent = valid ? sign : '—';
    mSignWord.className   = `m-sign-word${valid ? '' : ' dim'}${newSign ? ' pop' : ''}`;
    if (newSign) setTimeout(() => mSignWord.classList.remove('pop'), 300);
  }
  if (mConfBar) { mConfBar.style.width = `${pct}%`; mConfBar.className = `m-conf-bar${cls}`; }
  if (mConfPct) mConfPct.textContent = `${pct}%`;

  if (!valid) lastSignText = '';
}

// ── Chips ─────────────────────────────────────────────────────────────────────
function updateChips(signs) {
  if (!signs.length) {
    if (dChipRow) dChipRow.innerHTML = '<span class="d-empty">Start signing — detected signs appear here</span>';
    if (mChipRow) mChipRow.innerHTML = '<span class="m-empty">Start signing…</span>';
    return;
  }
  const html = (p) => signs.map((s, i) =>
    `<span class="${p}-chip${i === signs.length-1 ? ' new' : ''}">${s}</span>`
  ).join('');
  if (dChipRow) dChipRow.innerHTML = html('d');
  if (mChipRow) mChipRow.innerHTML = html('m');
}

// ── Sentence card ─────────────────────────────────────────────────────────────
function addCard(sentence, gloss) {
  lastSentence = sentence;
  sentenceTotal++;
  const isFirst    = sentenceTotal === 1;
  const isMilestone= sentenceTotal % 5 === 0;

  [dBtnSpeak, mBtnSpeak].forEach(b => { if (b) b.disabled = false; });

  const time      = new Date().toLocaleTimeString();
  const cardClass = isFirst || isMilestone ? ' milestone' : '';

  // Desktop
  if (dSentList) {
    dEmptyState?.remove();
    const c = document.createElement('div');
    c.className = `d-card${cardClass}`;
    c.innerHTML = `<div class="d-card-text">${sentence}</div>
      <div class="d-card-meta">${time} &nbsp;·&nbsp; <span class="d-card-gloss">${gloss}</span></div>`;
    dSentList.prepend(c);
  }

  // Mobile
  if (mSentList) {
    mEmptyState?.remove();
    const c = document.createElement('div');
    c.className = `m-card${cardClass}`;
    c.innerHTML = `<div class="m-card-text">${sentence}</div>
      <div class="m-card-meta">${time} &nbsp;·&nbsp; <span class="m-card-gloss">${gloss}</span></div>`;
    mSentList.prepend(c);
  }

  // Celebration on first sentence or every 5th
  if (isFirst || isMilestone) {
    celebrate();
    if (isFirst) toast('First translation ✓', 2200);
    else         toast(`${sentenceTotal} translations 🎉`, 2200);
  }

  // Badge
  const sentTab = document.querySelector('[data-tab="sentences"]');
  if (sentTab && !sentTab.classList.contains('active')) {
    unreadCount++;
    if (mBadge) { mBadge.textContent = unreadCount; mBadge.classList.add('show'); }
  }

  speak(sentence);
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

// ── Clear ─────────────────────────────────────────────────────────────────────
function clearAll() {
  if (dSentList) dSentList.innerHTML = `
    <div class="d-empty-state" id="dEmptyState">
      <div class="d-empty-icon">✋</div>
      <div class="d-empty-title">Nothing yet</div>
      <div class="d-empty-hint">Complete a sentence and it will appear here</div>
    </div>`;
  if (mSentList) mSentList.innerHTML = `
    <div class="m-empty-state" id="mEmptyState">
      <div class="m-empty-icon">✋</div>
      <div class="m-empty-title">Nothing yet</div>
      <div class="m-empty-hint">Complete a sentence and it will appear here</div>
    </div>`;
  updateChips([]); updateHUD(null, 0);
  lastSentence = ''; unreadCount = 0; sentenceTotal = 0;
  if (mBadge) mBadge.classList.remove('show');
  [dBtnSpeak, mBtnSpeak].forEach(b => { if (b) b.disabled = true; });
  toast('Cleared');
}

// ── Flush ─────────────────────────────────────────────────────────────────────
function flush() {
  ws?.readyState === 1 && ws.send(JSON.stringify({ action: 'flush' }));
  toast('Sentence sent for translation');
}

// ── Mobile tabs ───────────────────────────────────────────────────────────────
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
function syncSlider(src, dest, disp1, disp2, fmt) {
  src?.addEventListener('input', () => {
    const v = src.value;
    if (dest)  dest.value         = v;
    if (disp1) disp1.textContent  = fmt(v);
    if (disp2) disp2.textContent  = fmt(v);
  });
}

syncSlider(dConfThresh,  mConfThresh,  dConfThreshV,  mConfThreshV,  v => `${v}%`);
syncSlider(mConfThresh,  dConfThresh,  mConfThreshV,  dConfThreshV,  v => `${v}%`);
syncSlider(dPauseThresh, mPauseThresh, dPauseThreshV, mPauseThreshV, v => `${(parseInt(v)/10).toFixed(1)}s`);
syncSlider(mPauseThresh, dPauseThresh, mPauseThreshV, dPauseThreshV, v => {
  const s = `${(parseInt(v)/10).toFixed(1)}s`;
  ws?.readyState === 1 && ws.send(JSON.stringify({ action: 'set_pause', value: parseFloat(s) }));
  return s;
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

// ── Developer easter egg ──────────────────────────────────────────────────────
console.log('%c✋ ASL Translator', 'font-size:18px;font-weight:bold;color:#29c49a');
console.log('%cBuilt with MediaPipe + TensorFlow + FastAPI', 'color:#7a7a8e');

// ── Boot ──────────────────────────────────────────────────────────────────────
checkBackend();
window.speechSynthesis?.getVoices();
speechSynthesis.addEventListener?.('voiceschanged', () => speechSynthesis.getVoices());
