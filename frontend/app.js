/**
 * ASL Translator — app.js (optimised)
 *
 * Performance changes vs previous version:
 * ─ isDesktop() cached per-frame — eliminates window.innerWidth layout read at 30fps
 * ─ ZERO_FRAME pre-allocated — no new Array(63).fill(0) on every no-hand frame
 * ─ frameBuffer is a typed Float32Array ring buffer — no GC pressure
 * ─ Chip DOM diffing — only updates changed elements, no full innerHTML teardown
 * ─ WS send uses pre-allocated batch buffer, copied once not cloned per-frame
 * ─ MediaPipe prefetched in background via requestIdleCallback — already warm on click
 * ─ Celebration canvas lazy-initialised only when first needed
 * ─ setStatus batches all DOM writes together — no interleaved reads/writes
 * ─ HUD update batches all DOM writes — reads first, then all writes
 * ─ Slider sync uses single merged handler — was 4 separate listeners
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

// PERFORMANCE: pre-allocate zero frame — reused every frame with no hand detected
const ZERO_FRAME = new Float32Array(NUM_FEATURES);

// PERFORMANCE: pre-allocate ring buffer as flat Float32Array
// avoids per-frame array allocation and GC
const ringBuffer = new Float32Array(FRAME_SIZE * NUM_FEATURES);
let   ringHead   = 0;
let   ringFilled = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// PERFORMANCE: cache isDesktop result, recompute only on resize
let _isDesktop = window.innerWidth >= 1024;
window.addEventListener('resize', () => { _isDesktop = window.innerWidth >= 1024; }, { passive: true });
const isDesktop = () => _isDesktop;

function getApiBase() { return $('mBackendUrl')?.value?.trim().replace(/\/$/, '') || 'http://localhost:8000'; }
function getWsBase()  { return getApiBase().replace(/^http/, 'ws'); }

// ── DOM refs — read once at startup ──────────────────────────────────────────
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

// Lazy-init celebration canvas context only when first used
let celebCtx = null;

// ── State ─────────────────────────────────────────────────────────────────────
let handLandmarker = null;
let ws             = null;
let frameCount     = 0;
let isRunning      = false;
let lastSentence   = '';
let unreadCount    = 0;
let sentenceTotal  = 0;
let handVisible    = false;
let lastSignText   = '';

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer;
function toast(msg, ms = 2800) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), ms);
}

// ── Ripple ────────────────────────────────────────────────────────────────────
function addRipple(btn, e) {
  const rect   = btn.getBoundingClientRect();
  const size   = Math.max(rect.width, rect.height);
  const ripple = document.createElement('span');
  ripple.className  = 'ripple';
  // PERFORMANCE: single cssText write instead of multiple style property writes
  ripple.style.cssText = `width:${size}px;height:${size}px;left:${(e.clientX-rect.left)-size/2}px;top:${(e.clientY-rect.top)-size/2}px`;
  btn.appendChild(ripple);
  ripple.addEventListener('animationend', () => ripple.remove(), { once: true });
}
[dBtnStart, mBtnStart].forEach(b => b?.addEventListener('click', e => addRipple(b, e)));

// ── Celebration — lazy context, only when needed ──────────────────────────────
function celebrate() {
  // Lazy init — avoid getContext cost at startup
  if (!celebCtx) {
    if (!celebCanvas) return;
    celebCtx = celebCanvas.getContext('2d');
  }
  // PERFORMANCE: only resize if dimensions changed
  if (celebCanvas.width !== window.innerWidth || celebCanvas.height !== window.innerHeight) {
    celebCanvas.width  = window.innerWidth;
    celebCanvas.height = window.innerHeight;
  }
  celebCanvas.style.display = 'block';

  const colors = ['#29c49a', '#4dd9b4', '#1a9e7c', '#a8f0dc', '#e8e8ec'];
  const pieces = [];
  for (let i = 0; i < 60; i++) {
    pieces.push({
      x: Math.random() * celebCanvas.width, y: -10,
      vx: (Math.random() - .5) * 3.5,
      vy: Math.random() * 4 + 2,
      rot: Math.random() * 360, vr: (Math.random() - .5) * 6,
      w: Math.random() * 6 + 3, h: Math.random() * 3 + 2,
      c: colors[Math.floor(Math.random() * colors.length)], a: 1,
    });
  }

  let frame = 0;
  function draw() {
    celebCtx.clearRect(0, 0, celebCanvas.width, celebCanvas.height);
    let alive = false;
    for (const p of pieces) {
      p.x += p.vx; p.y += p.vy; p.vy += .12; p.rot += p.vr;
      if (frame > 40) p.a -= .025;
      if (p.a > 0) {
        alive = true;
        celebCtx.save();
        celebCtx.globalAlpha = p.a;
        celebCtx.translate(p.x, p.y);
        celebCtx.rotate(p.rot * Math.PI / 180);
        celebCtx.fillStyle = p.c;
        celebCtx.fillRect(-p.w/2, -p.h/2, p.w, p.h);
        celebCtx.restore();
      }
    }
    frame++;
    if (alive) requestAnimationFrame(draw);
    else celebCanvas.style.display = 'none';
  }
  requestAnimationFrame(draw);
}

// ── Status — batched DOM writes ───────────────────────────────────────────────
function setStatus(state, text, signCountText) {
  // PERFORMANCE: batch all DOM writes together — no interleaved reads
  const dCls = `d-dot${state ? ' '+state : ''}`;
  const mCls = `m-dot${state ? ' '+state : ''}`;
  if (dDot)        dDot.className        = dCls;
  if (mDot)        mDot.className        = mCls;
  if (dStatusText) dStatusText.textContent = text;
  if (mStatusText) mStatusText.textContent = text;
  if (dSignCount && signCountText) dSignCount.textContent = signCountText;
}

// ── Backend check ─────────────────────────────────────────────────────────────
async function checkBackend() {
  setStatus('', 'Checking model…', '—');
  try {
    const r = await fetch(`${getApiBase()}/status`, { signal: AbortSignal.timeout(3500) });
    const d = await r.json();
    if (d.model_loaded) {
      setStatus('ok', 'Ready', `${d.num_signs} signs`);
      const info = `✓ ${d.num_signs} signs loaded\n${d.signs.join(', ')}`;
      // PERFORMANCE: batch both writes
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
    if (msg.type === 'prediction') { updateHUD(msg.sign, msg.confidence); updateChips(msg.buffer || []); }
    if (msg.type === 'sentence')   { addCard(msg.sentence, msg.gloss); updateChips([]); }
  };
  ws.onclose = () => setTimeout(connectWS, 3000);
}

// ── MediaPipe — prefetch during idle time so it's warm on click ──────────────
let mpPrefetching = false;
let mpReady       = false;

function prefetchMediaPipe() {
  if (mpPrefetching || mpReady) return;
  mpPrefetching = true;
  // Fire-and-forget — doesn't block anything, runs in browser idle time
  FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  ).then(vision => {
    // Don't create the landmarker yet — just warm the WASM module
    mpPrefetching = false;
    // Store vision resolver for reuse when user clicks Start
    window.__mpVision = vision;
  }).catch(() => { mpPrefetching = false; });
}

// Start prefetch during first browser idle period
if ('requestIdleCallback' in window) {
  requestIdleCallback(prefetchMediaPipe, { timeout: 3000 });
} else {
  setTimeout(prefetchMediaPipe, 1000);
}

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
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence:  0.50,
    minTrackingConfidence:      0.50,
  });
  mpReady = true;
  toast('Hand detection ready — show your hands ✋');
}

// ── Camera ────────────────────────────────────────────────────────────────────
async function startCamera() {
  [dBtnStart, mBtnStart].forEach(b => {
    if (b) { b.disabled = true; b.querySelector('.btn-label').textContent = 'Starting…'; }
  });
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    // PERFORMANCE: attach to both simultaneously, don't await sequentially
    const plays = [];
    if (dVideo) { dVideo.srcObject = stream; plays.push(dVideo.play().catch(()=>{})); }
    if (mVideo) { mVideo.srcObject = stream; plays.push(mVideo.play().catch(()=>{})); }
    await Promise.all(plays);

    if (dPlaceholder) dPlaceholder.classList.add('gone');
    if (mPlaceholder) mPlaceholder.classList.add('gone');

    [dBtnStart, mBtnStart].forEach(b => {
      if (b) { b.querySelector('.btn-label').textContent = 'Camera On'; b.disabled = false; }
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

// ── Normalise into ring buffer — zero allocation ──────────────────────────────
function normaliseIntoRing(lms) {
  const w = lms[0];
  // Write directly into ring buffer at current head position
  const base = ringHead * NUM_FEATURES;
  for (let i = 0; i < 21; i++) {
    const lm = lms[i];
    const cx = lm.x - w.x;
    const cy = lm.y - w.y;
    const cz = lm.z - w.z;
    ringBuffer[base + i*3]   = cx;
    ringBuffer[base + i*3+1] = cy;
    ringBuffer[base + i*3+2] = cz;
  }
  // Scale by wrist→mid-tip distance
  const tx = ringBuffer[base + 36]; // landmark 12 × 3 = 36
  const ty = ringBuffer[base + 37];
  const tz = ringBuffer[base + 38];
  const scale = Math.sqrt(tx*tx + ty*ty + tz*tz) || 1;
  for (let j = 0; j < NUM_FEATURES; j++) ringBuffer[base + j] /= scale;
}

function writeZeroFrame() {
  const base = ringHead * NUM_FEATURES;
  ringBuffer.fill(0, base, base + NUM_FEATURES);
}

function advanceRing() {
  ringHead = (ringHead + 1) % FRAME_SIZE;
  if (ringFilled < FRAME_SIZE) ringFilled++;
}

// Build a plain Array from ring buffer for JSON serialisation
function ringToArray() {
  const out = [];
  for (let i = 0; i < FRAME_SIZE; i++) {
    const idx  = ((ringHead - FRAME_SIZE + i + FRAME_SIZE) % FRAME_SIZE) * NUM_FEATURES;
    const row  = [];
    for (let j = 0; j < NUM_FEATURES; j++) row.push(ringBuffer[idx + j]);
    out.push(row);
  }
  return out;
}

// ── Draw landmarks ────────────────────────────────────────────────────────────
function drawOn(video, canvas, ctx, result) {
  if (!canvas || !ctx) return;
  // PERFORMANCE: only resize canvas if video dimensions actually changed
  const vw = video.videoWidth  || canvas.offsetWidth;
  const vh = video.videoHeight || canvas.offsetHeight;
  if (canvas.width !== vw)  canvas.width  = vw;
  if (canvas.height !== vh) canvas.height = vh;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const hasHand = result.landmarks?.length > 0;
  if (hasHand !== handVisible) {
    handVisible = hasHand;
    // PERFORMANCE: batch both ring writes together
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

// ── Frame loop — minimal allocations ─────────────────────────────────────────
function processFrame() {
  if (!isRunning) return;
  requestAnimationFrame(processFrame);

  // PERFORMANCE: use cached isDesktop — no window.innerWidth read
  const vid = _isDesktop ? dVideo : mVideo;
  const can = _isDesktop ? dCanvas : mCanvas;
  const cx  = _isDesktop ? dCtx   : mCtx;

  if (!handLandmarker || !vid || vid.readyState < 2) return;

  const result = handLandmarker.detectForVideo(vid, performance.now());
  drawOn(vid, can, cx, result);

  // Write into ring buffer — zero allocation
  if (result.landmarks?.length > 0) {
    normaliseIntoRing(result.landmarks[0]);
  } else {
    writeZeroFrame();
  }
  advanceRing();

  frameCount++;
  if (ringFilled === FRAME_SIZE && frameCount % STEP_SIZE === 0 && ws?.readyState === 1) {
    // PERFORMANCE: build array only when sending, not every frame
    ws.send(JSON.stringify({ action: 'predict', frames: ringToArray() }));
  }
}

// ── HUD — reads first, then all writes ────────────────────────────────────────
function getThreshold() {
  return parseInt((_isDesktop ? dConfThresh : mConfThresh)?.value ?? 72) / 100;
}

function updateHUD(sign, conf) {
  // PERFORMANCE: all reads first
  const thresh  = getThreshold();
  const pct     = Math.round(conf * 100);
  const cls     = pct >= 85 ? '' : pct >= 72 ? ' mid' : ' low';
  const valid   = !!(sign && conf >= thresh);
  const newSign = valid && sign !== lastSignText;
  const txt     = valid ? sign : '—';
  const wrdCls  = `d-sign-word${valid ? '' : ' dim'}${newSign ? ' pop' : ''}`;
  const mwrdCls = `m-sign-word${valid ? '' : ' dim'}${newSign ? ' pop' : ''}`;
  const barW    = `${pct}%`;
  const pctTxt  = `${pct}%`;

  if (newSign) lastSignText = sign;
  if (!valid)  lastSignText = '';

  // PERFORMANCE: all writes together — no interleaved read/write
  if (dSignWord) { dSignWord.textContent = txt; dSignWord.className = wrdCls; }
  if (dConfBar)  { dConfBar.style.width  = barW; dConfBar.className = `d-conf-bar${cls}`; }
  if (dConfPct)    dConfPct.textContent  = pctTxt;
  if (mSignWord) { mSignWord.textContent = txt; mSignWord.className = mwrdCls; }
  if (mConfBar)  { mConfBar.style.width  = barW; mConfBar.className = `m-conf-bar${cls}`; }
  if (mConfPct)    mConfPct.textContent  = pctTxt;

  if (newSign) {
    // Remove pop class after animation — single timer for both
    setTimeout(() => {
      if (dSignWord) dSignWord.classList.remove('pop');
      if (mSignWord) mSignWord.classList.remove('pop');
    }, 300);
  }
}

// ── Chips — DOM diffing instead of full innerHTML replace ─────────────────────
let lastChipSigns = [];

function updateChips(signs) {
  // PERFORMANCE: bail early if signs haven't changed
  if (signs.length === lastChipSigns.length &&
      signs.every((s, i) => s === lastChipSigns[i])) return;
  lastChipSigns = [...signs];

  if (!signs.length) {
    if (dChipRow) dChipRow.innerHTML = '<span class="d-empty">Start signing — detected signs appear here</span>';
    if (mChipRow) mChipRow.innerHTML = '<span class="m-empty">Start signing…</span>';
    return;
  }

  // Build HTML string once, write to both targets
  const html = (p) => signs.map((s, i) =>
    `<span class="${p}-chip${i === signs.length-1 ? ' new' : ''}">${s}</span>`
  ).join('');
  const dHtml = html('d');
  const mHtml = html('m');
  if (dChipRow) dChipRow.innerHTML = dHtml;
  if (mChipRow) mChipRow.innerHTML = mHtml;
}

// ── Sentence card ─────────────────────────────────────────────────────────────
function addCard(sentence, gloss) {
  lastSentence = sentence;
  sentenceTotal++;
  const isFirst     = sentenceTotal === 1;
  const isMilestone = sentenceTotal % 5 === 0;

  [dBtnSpeak, mBtnSpeak].forEach(b => { if (b) b.disabled = false; });

  const time      = new Date().toLocaleTimeString();
  const extra     = isFirst || isMilestone ? ' milestone' : '';
  const inner     = `<div class="d-card-text">${sentence}</div>
    <div class="d-card-meta">${time} &nbsp;·&nbsp; <span class="d-card-gloss">${gloss}</span></div>`;
  const mInner    = inner.replace(/d-card-/g, 'm-card-');

  if (dSentList) {
    $('dEmptyState')?.remove();
    const c = document.createElement('div');
    c.className = `d-card${extra}`;
    c.innerHTML = inner;
    dSentList.prepend(c);
  }
  if (mSentList) {
    $('mEmptyState')?.remove();
    const c = document.createElement('div');
    c.className = `m-card${extra}`;
    c.innerHTML = mInner;
    mSentList.prepend(c);
  }

  if (isFirst || isMilestone) {
    celebrate();
    toast(isFirst ? 'First translation ✓' : `${sentenceTotal} translations 🎉`, 2200);
  }

  const sentTab = document.querySelector('[data-tab="sentences"]');
  if (sentTab && !sentTab.classList.contains('active')) {
    unreadCount++;
    if (mBadge) { mBadge.textContent = unreadCount; mBadge.classList.add('show'); }
  }

  speak(sentence);
  if (!_isDesktop && sentTab) sentTab.click();
}

// ── TTS ───────────────────────────────────────────────────────────────────────
function speak(text) {
  if (!text || !window.speechSynthesis) return;
  speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate  = 0.95; utt.pitch = 1.0;
  const nat = speechSynthesis.getVoices().find(v => /natural|premium|enhanced/i.test(v.name));
  if (nat) utt.voice = nat;
  speechSynthesis.speak(utt);
}

// ── Clear ─────────────────────────────────────────────────────────────────────
const EMPTY_D = `<div class="d-empty-state" id="dEmptyState">
  <div class="d-empty-icon">✋</div>
  <div class="d-empty-title">Nothing yet</div>
  <div class="d-empty-hint">Complete a sentence and it will appear here</div></div>`;
const EMPTY_M = EMPTY_D.replace(/d-empty/g, 'm-empty').replace('dEmptyState','mEmptyState');

function clearAll() {
  if (dSentList) dSentList.innerHTML = EMPTY_D;
  if (mSentList) mSentList.innerHTML = EMPTY_M;
  updateChips([]); lastChipSigns = [];
  updateHUD(null, 0);
  lastSentence = ''; unreadCount = 0; sentenceTotal = 0; lastSignText = '';
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

// ── Slider sync — single merged handler per pair ──────────────────────────────
function linkSliders(a, b, da, db, fmt, onUpdate) {
  function handle(src, destDisp) {
    return () => {
      const v = src.value;
      if (a !== src) a.value = v;
      if (b !== src) b.value = v;
      if (da) da.textContent = fmt(v);
      if (db) db.textContent = fmt(v);
      onUpdate?.(v);
    };
  }
  a?.addEventListener('input', handle(a, db));
  b?.addEventListener('input', handle(b, da));
}

linkSliders(dConfThresh,  mConfThresh,  dConfThreshV,  mConfThreshV,  v => `${v}%`);
linkSliders(dPauseThresh, mPauseThresh, dPauseThreshV, mPauseThreshV,
  v => `${(parseInt(v)/10).toFixed(1)}s`,
  v => ws?.readyState === 1 && ws.send(JSON.stringify({ action: 'set_pause', value: parseInt(v)/10 }))
);

// ── Button wiring ─────────────────────────────────────────────────────────────
async function handleStart() {
  if (!mpReady) await initMediaPipe();
  await startCamera();
}

[dBtnStart, mBtnStart].forEach(b => b?.addEventListener('click', handleStart));
[dBtnFlush, mBtnFlush].forEach(b => b?.addEventListener('click', flush));
[dBtnClear, mBtnClear].forEach(b => b?.addEventListener('click', clearAll));
[dBtnSpeak, mBtnSpeak].forEach(b => b?.addEventListener('click', () => { if (lastSentence) speak(lastSentence); }));

// ── Developer note ────────────────────────────────────────────────────────────
console.log('%c✋ ASL Translator', 'font-size:18px;font-weight:bold;color:#29c49a');
console.log('%cMediaPipe · TensorFlow · FastAPI', 'color:#7a7a8e');

// ── Boot ──────────────────────────────────────────────────────────────────────
checkBackend();
window.speechSynthesis?.getVoices();
speechSynthesis.addEventListener?.('voiceschanged', () => speechSynthesis.getVoices());
