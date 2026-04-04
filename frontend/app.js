/**
 * ASL Translator — app.js
 * Camera → MediaPipe Holistic (543 landmarks) → WebSocket → TFLite → Sign prediction
 */

// ── CDN urls ──────────────────────────────────────────────────────────────────
const HOLISTIC_SCRIPT = "https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/holistic.js";
const HOLISTIC_UTILS  = "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3.1675466862/camera_utils.js";
const DRAWING_UTILS   = "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1675466124/drawing_utils.js";

// ── Constants ─────────────────────────────────────────────────────────────────
const CLIENT_ID         = Math.random().toString(36).slice(2);
const CONTROLS_TIMEOUT  = 3000;
const SEND_EVERY_N      = 15;      // trigger predict every N frames (~0.5s at 30fps)
const WS_RECONNECT_BASE = 1000;
const WS_RECONNECT_MAX  = 16000;

/** Pre-serialized string — avoids JSON.stringify on every predict trigger */
const PREDICT_MSG = '{"action":"predict"}';

/** Reuse frozen style objects — MediaPipe drawing never allocates per frame */
const DRAW_CONN = Object.freeze({ color: 'rgba(41,196,154,.35)', lineWidth: 1.5 });
const DRAW_LM   = Object.freeze({ color: 'rgba(41,196,154,.8)',  lineWidth: 1, radius: 3 });

const CONFETTI_COLORS = ['#29c49a', '#4dd9b4', '#1a9e7c', '#a8f0dc', '#e8e8ec'];

// ── Helpers ───────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function getApiBase() {
  return $('backendUrl')?.value?.trim().replace(/\/$/, '') || 'http://localhost:8000';
}
function getWsBase() {
  return getApiBase().replace(/^https/, 'wss').replace(/^http/, 'ws');
}

/** Safe send — only fires when socket is OPEN */
function wsSend(payload) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(payload);
}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const video            = $('video');
const canvas           = $('overlay');
const ctx              = canvas?.getContext('2d');
const startScreen      = $('startScreen');
const topBar           = $('topBar');
const topDot           = $('topDot');
const topStatus        = $('topStatus');
const handRing         = $('handRing');
const signWord         = $('signWord');
const sentenceText     = $('sentenceText');
const sentenceChips    = $('sentenceChips');
const controls         = $('controls');
const settingsPanel    = $('settingsPanel');
const historyPanel     = $('historyPanel');
const historyList      = $('historyList');
const modelInfo        = $('modelInfo');
const statusDot        = $('statusDot');
const statusText       = $('statusText');
const btnStart         = $('btnStart');
const btnFlush         = $('btnFlush');
const btnSpeak         = $('btnSpeak');
const btnClear         = $('btnClear');
const btnSettings      = $('btnSettings');
const btnSettingsClose = $('btnSettingsClose');
const btnHistoryClose  = $('btnHistoryClose');
const confThresh       = $('confThresh');
const confThreshVal    = $('confThreshVal');
const pauseThresh      = $('pauseThresh');
const pauseThreshVal   = $('pauseThreshVal');
const toastEl          = $('toast');
const celebCanvas      = $('celebrationCanvas');
const gateLabel        = $('gateLabel');      // optional: gate reason
const marginLabel      = $('marginLabel');    // optional: margin value
const top5Panel        = $('top5Panel');      // optional: top-5 panel
const suggestionsBar   = $('suggestionsBar'); // optional: autocomplete

let celebCtx         = null;
let ws               = null;
let holistic         = null;
let holisticCamera   = null;
let isRunning        = false;
let lastSentence     = '';
let sentenceTotal    = 0;
let handVisible      = false;
let lastSignText     = '';
let lastChipSigns    = [];
let controlsTimer    = null;
let frameCount       = 0;
let lastVideoW       = 0;
let lastVideoH       = 0;
let wsReconnectDelay = WS_RECONNECT_BASE;
let hudConfThresh    = 0.5; // cached from slider — avoids DOM read per WS message

// ── Script loader ─────────────────────────────────────────────────────────────
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src; s.onload = resolve; s.onerror = reject;
    document.head.appendChild(s);
  });
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer;
function toast(msg, ms = 2600) {
  if (!toastEl) return;
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
  r.style.cssText = `width:${size}px;height:${size}px;left:${(e.clientX - rect.left) - size/2}px;top:${(e.clientY - rect.top) - size/2}px`;
  btn.appendChild(r);
  r.addEventListener('animationend', () => r.remove(), { once: true });
}
btnStart?.addEventListener('click', e => addRipple(btnStart, e));

// ── Controls auto-hide ────────────────────────────────────────────────────────
function showControls() {
  controls?.classList.remove('hidden');
  topBar?.classList.remove('hidden');
  clearTimeout(controlsTimer);
  controlsTimer = setTimeout(() => {
    controls?.classList.add('hidden');
    topBar?.classList.add('hidden');
  }, CONTROLS_TIMEOUT);
}

document.addEventListener('pointerdown', (e) => {
  if (e.target.closest('.settings-panel, .history-panel, .start-screen')) return;
  if (isRunning) showControls();
}, { passive: true });

// ── Status ────────────────────────────────────────────────────────────────────
function setStatus(state, text) {
  const cls = `sdot${state ? ' ' + state : ''}`;
  if (statusDot) statusDot.className = cls;
  if (topDot)    topDot.className    = cls;
  if (statusText) statusText.textContent = text;
  if (topStatus)  topStatus.textContent  = text;
}

// ── Backend check ─────────────────────────────────────────────────────────────
async function checkBackend() {
  setStatus('', 'Checking…');
  try {
    const r = await fetch(`${getApiBase()}/status`, { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    if (d.model_loaded) {
      setStatus('ok', `${d.num_signs} signs`);
      const signs = Array.isArray(d.signs) ? d.signs : [];
      if (modelInfo) modelInfo.textContent =
        `✓ ${d.num_signs} signs loaded\nModel: ${d.model || 'hoyso48'}\n${signs.slice(0, 10).join(', ')}…`;
    } else {
      setStatus('warn', 'No model');
      if (modelInfo) modelInfo.textContent = `⚠ No model\n${d.error || 'Copy model files to backend/model/'}`;
      toast('⚠ No model found', 5000);
    }
  } catch {
    setStatus('error', 'Offline');
    if (modelInfo) modelInfo.textContent = '✕ Backend offline\nuvicorn main:app --reload';
    toast('✕ Backend offline — start the server first', 5000);
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS() {
  if (ws && ws.readyState <= WebSocket.OPEN) return;

  ws = new WebSocket(`${getWsBase()}/ws/${CLIENT_ID}`);

  ws.onopen = () => {
    setStatus('ok', 'Live');
    wsReconnectDelay = WS_RECONNECT_BASE;
    // Sync confidence slider to backend on reconnect
    wsSend(JSON.stringify({ action: 'set_threshold', confidence: hudConfThresh }));
  };

  ws.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); }
    catch { console.warn('[WS] Bad JSON:', e.data); return; }

    if (msg.type === 'prediction') {
      updateHUD(msg.sign, msg.confidence);
      updateChips(msg.buffer || []);
      updateDiagHUD(msg);
    }
    if (msg.type === 'sentence') {
      addSentence(msg.sentence, msg.gloss);
      updateChips([]);
    }
  };

  ws.onerror = () => setStatus('warn', 'WS error');

  ws.onclose = () => {
    setStatus('warn', 'Reconnecting…');
    setTimeout(connectWS, wsReconnectDelay);
    wsReconnectDelay = Math.min(wsReconnectDelay * 2, WS_RECONNECT_MAX);
  };
}

// ── Diagnostic HUD ────────────────────────────────────────────────────────────
function updateDiagHUD(msg) {
  if (gateLabel) {
    gateLabel.textContent = msg.gate ? `⛔ ${msg.gate.replace(/_/g, ' ')}` : '';
  }
  if (marginLabel && msg.margin != null) {
    marginLabel.textContent = `margin: ${(msg.margin * 100).toFixed(1)}%`;
  }
  if (top5Panel && Array.isArray(msg.top5) && msg.top5.length) {
    top5Panel.innerHTML = msg.top5
      .map(p => `<span class="t5-chip" title="${(p.confidence * 100).toFixed(1)}%">
        ${p.sign} <em>${(p.confidence * 100).toFixed(0)}%</em></span>`)
      .join('');
  }
  if (suggestionsBar) {
    if (Array.isArray(msg.suggestions) && msg.suggestions.length) {
      suggestionsBar.innerHTML = msg.suggestions
        .map(s => `<button class="sugg-chip" data-sign="${s.sign}">${s.sign}</button>`)
        .join('');
      suggestionsBar.querySelectorAll('.sugg-chip').forEach(btn => {
        btn.addEventListener('click', () => {
          wsSend(JSON.stringify({ action: 'accept_suggestion', sign: btn.dataset.sign }));
          toast(`✓ Added: ${btn.dataset.sign}`);
        }, { once: true });
      });
    } else {
      suggestionsBar.innerHTML = '';
    }
  }
}

// ── MediaPipe Holistic ────────────────────────────────────────────────────────
async function initHolistic() {
  toast('Loading hand detection…');
  await loadScript(HOLISTIC_UTILS);
  await loadScript(DRAWING_UTILS);
  await loadScript(HOLISTIC_SCRIPT);

  holistic = new window.Holistic({
    locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${f}`
  });

  holistic.setOptions({
    modelComplexity:        1,
    smoothLandmarks:        true,
    enableSegmentation:     false,
    smoothSegmentation:     false,
    refineFaceLandmarks:    false,  // saves GPU, not needed for sign recognition
    minDetectionConfidence: 0.5,
    minTrackingConfidence:  0.5,
  });

  holistic.onResults(onHolisticResults);
  toast('Ready — show your hands ✋');
}

// ── Pack landmarks ────────────────────────────────────────────────────────────
/**
 * FIX: Always include ALL four landmark groups on every frame.
 *
 * Previous version omitted faceLandmarks from regular frames to save bandwidth,
 * but the backend nose-centres every frame using face landmark #1.
 * Without face, nose = (0,0,0) so no centring happened on 14 out of every 15
 * frames — producing an inconsistent feature window going into the model.
 *
 * Bandwidth cost: face adds ~468 × 3 floats ≈ 11KB/frame at 30fps = ~330KB/s.
 * This is acceptable on a local WebSocket. If bandwidth becomes a concern,
 * implement centring in the frontend instead and strip face before sending.
 */
function packLandmarks(res) {
  const o = {};
  if (res.faceLandmarks)      o.faceLandmarks      = res.faceLandmarks;
  if (res.poseLandmarks)      o.poseLandmarks      = res.poseLandmarks;
  if (res.leftHandLandmarks)  o.leftHandLandmarks  = res.leftHandLandmarks;
  if (res.rightHandLandmarks) o.rightHandLandmarks = res.rightHandLandmarks;
  return o;
}

// ── Holistic results handler ──────────────────────────────────────────────────
function onHolisticResults(results) {
  if (!canvas || !ctx || !video) return;

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (vw > 0 && vh > 0 && (vw !== lastVideoW || vh !== lastVideoH)) {
    lastVideoW = vw; lastVideoH = vh;
    canvas.width = vw; canvas.height = vh;
  }
  if (vw <= 0 || vh <= 0) return;
  ctx.clearRect(0, 0, vw, vh);

  const hasHand = (results.leftHandLandmarks?.length > 0)
               || (results.rightHandLandmarks?.length > 0);
  if (hasHand !== handVisible) {
    handVisible = hasHand;
    handRing?.classList.toggle('active', hasHand);
  }

  if (results.leftHandLandmarks) {
    window.drawConnectors(ctx, results.leftHandLandmarks, window.HAND_CONNECTIONS, DRAW_CONN);
    window.drawLandmarks(ctx,  results.leftHandLandmarks, DRAW_LM);
  }
  if (results.rightHandLandmarks) {
    window.drawConnectors(ctx, results.rightHandLandmarks, window.HAND_CONNECTIONS, DRAW_CONN);
    window.drawLandmarks(ctx,  results.rightHandLandmarks, DRAW_LM);
  }

  if (ws?.readyState !== WebSocket.OPEN) return;

  frameCount++;

  // Send frame with ALL landmarks (face included) every frame
  ws.send(JSON.stringify({ action: 'frame', landmarks: packLandmarks(results) }));

  if (frameCount >= SEND_EVERY_N) {
    ws.send(PREDICT_MSG);
    frameCount = 0;
  }
}

// ── Start camera ──────────────────────────────────────────────────────────────
async function startCamera() {
  btnStart.disabled = true;
  btnStart.querySelector('.btn-label').textContent = 'Starting…';

  try {
    if (!holistic) await initHolistic();

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    holisticCamera = new window.Camera(video, {
      onFrame: async () => { if (holistic) await holistic.send({ image: video }); },
      width: 1280, height: 720,
    });
    holisticCamera.start();

    startScreen?.classList.add('gone');
    if (btnFlush) btnFlush.disabled = false;
    isRunning = true;
    connectWS();
    showControls();

    btnStart.querySelector('.btn-label').textContent = 'Camera On';
    btnStart.disabled = false;
  } catch (err) {
    btnStart.disabled = false;
    btnStart.querySelector('.btn-label').textContent = 'Start Camera';
    toast(`✕ Camera error: ${err.message}`, 5000);
  }
}

// ── HUD ───────────────────────────────────────────────────────────────────────
function updateHUD(sign, conf) {
  const valid   = !!(sign && conf >= hudConfThresh);
  const newSign = valid && sign !== lastSignText;
  lastSignText  = valid ? sign : '';

  if (signWord) {
    signWord.textContent = valid ? sign : '';
    signWord.className   = `sign-word${valid ? '' : ' dim'}${newSign ? ' pop' : ''}`;
    if (newSign) setTimeout(() => signWord.classList.remove('pop'), 280);
  }
}

function updateChips(signs) {
  if (signs.length === lastChipSigns.length &&
      signs.every((s, i) => s === lastChipSigns[i])) return;
  lastChipSigns = [...signs];
  if (!signs.length) { if (sentenceChips) sentenceChips.innerHTML = ''; return; }
  if (sentenceChips) sentenceChips.innerHTML = signs
    .map((s, i) => `<span class="s-chip${i === signs.length - 1 ? ' new' : ''}">${s}</span>`)
    .join('');
}

// ── Sentence ──────────────────────────────────────────────────────────────────
function addSentence(sentence, gloss) {
  lastSentence = sentence;
  sentenceTotal++;
  const isFirst     = sentenceTotal === 1;
  const isMilestone = sentenceTotal % 5 === 0;

  if (btnSpeak) btnSpeak.disabled = false;
  if (sentenceText) {
    sentenceText.textContent = sentence;
    sentenceText.classList.remove('placeholder');
  }

  historyList?.querySelector('.history-empty')?.remove();
  const card = document.createElement('div');
  card.className = 'h-card';
  card.innerHTML = `<div class="h-card-text">${sentence}</div>
    <div class="h-card-meta">${new Date().toLocaleTimeString()} &nbsp;·&nbsp; <span class="h-card-gloss">${gloss}</span></div>`;
  historyList?.prepend(card);

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
  if (!celebCanvas) return;
  if (!celebCtx) celebCtx = celebCanvas.getContext('2d');
  if (celebCanvas.width  !== window.innerWidth)  celebCanvas.width  = window.innerWidth;
  if (celebCanvas.height !== window.innerHeight) celebCanvas.height = window.innerHeight;
  celebCanvas.style.display = 'block';

  const pieces = Array.from({ length: 60 }, () => ({
    x: Math.random() * celebCanvas.width, y: -10,
    vx: (Math.random() - .5) * 3.5, vy: Math.random() * 4 + 2,
    rot: Math.random() * 360, vr: (Math.random() - .5) * 6,
    w: Math.random() * 6 + 3, h: Math.random() * 3 + 2,
    c: CONFETTI_COLORS[Math.floor(Math.random() * CONFETTI_COLORS.length)], a: 1,
  }));

  let frame = 0;
  (function draw() {
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
        celebCtx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
        celebCtx.restore();
      }
    }
    frame++;
    if (alive) requestAnimationFrame(draw);
    else celebCanvas.style.display = 'none';
  })();
}

// ── Clear ─────────────────────────────────────────────────────────────────────
function clearAll() {
  if (sentenceText) {
    sentenceText.textContent = 'Completed sentences appear here';
    sentenceText.classList.add('placeholder');
  }
  if (sentenceChips) sentenceChips.innerHTML = '';
  updateHUD(null, 0);
  lastSentence = ''; sentenceTotal = 0; lastSignText = ''; lastChipSigns = [];
  if (historyList) historyList.innerHTML = '<div class="history-empty">No translations yet</div>';
  if (btnSpeak) btnSpeak.disabled = true;
  toast('Cleared');
}

// ── Flush ─────────────────────────────────────────────────────────────────────
function flush() {
  wsSend(JSON.stringify({ action: 'flush' }));
  toast('Sentence flushed');
}

// ── Panels ────────────────────────────────────────────────────────────────────
btnSettings?.addEventListener('click', () => {
  settingsPanel?.classList.add('open');
  clearTimeout(controlsTimer);
});
btnSettingsClose?.addEventListener('click', () => {
  settingsPanel?.classList.remove('open');
  showControls();
});
settingsPanel?.addEventListener('click', e => {
  if (e.target === settingsPanel) settingsPanel.classList.remove('open');
});
sentenceText?.addEventListener('click', () => {
  if (sentenceTotal > 0) historyPanel?.classList.add('open');
});
btnHistoryClose?.addEventListener('click', () => historyPanel?.classList.remove('open'));
historyPanel?.addEventListener('click', e => {
  if (e.target === historyPanel) historyPanel.classList.remove('open');
});

// ── Sliders ───────────────────────────────────────────────────────────────────
confThresh?.addEventListener('input', () => {
  hudConfThresh = parseInt(confThresh.value, 10) / 100;
  if (confThreshVal) confThreshVal.textContent = `${confThresh.value}%`;
  wsSend(JSON.stringify({ action: 'set_threshold', confidence: hudConfThresh }));
});

pauseThresh?.addEventListener('input', () => {
  const val = (parseInt(pauseThresh.value) / 10).toFixed(1);
  if (pauseThreshVal) pauseThreshVal.textContent = `${val}s`;
  wsSend(JSON.stringify({ action: 'set_pause', value: parseFloat(val) }));
});

// ── Button wiring ─────────────────────────────────────────────────────────────
btnStart?.addEventListener('click', startCamera);
btnFlush?.addEventListener('click', flush);
btnClear?.addEventListener('click', clearAll);
btnSpeak?.addEventListener('click', () => { if (lastSentence) speak(lastSentence); });

// ── Boot ──────────────────────────────────────────────────────────────────────
if (sentenceText) sentenceText.classList.add('placeholder');
if (confThresh)   hudConfThresh = parseInt(confThresh.value, 10) / 100;
checkBackend();
window.speechSynthesis?.getVoices();
speechSynthesis.addEventListener?.('voiceschanged', () => speechSynthesis.getVoices());
console.log('%c✋ ASL Translator', 'font-size:18px;font-weight:bold;color:#29c49a');
console.log('%cPowered by hoyso48 TFLite model (Kaggle 1st place)', 'color:#7a7a8e');