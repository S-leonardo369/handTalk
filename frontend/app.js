/**
 * ASL Translator — app.js (Holistic landmark version)
 * Uses MediaPipe Holistic to extract 543 landmarks (face + hands + pose)
 * and sends them to the FastAPI backend which runs the hoyso48 TFLite model.
 *
 * Pipeline:
 *   Camera → Holistic (543 landmarks) → WebSocket → TFLite → Sign prediction
 */

// ── MediaPipe Holistic (CDN — works without local files) ──────────────────────
// We load Holistic from CDN since it is a different model than HandLandmarker
const HOLISTIC_SCRIPT = "https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/holistic.js";
const HOLISTIC_UTILS  = "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3.1675466862/camera_utils.js";
const DRAWING_UTILS   = "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1675466124/drawing_utils.js";

// ── Constants ─────────────────────────────────────────────────────────────────
const CLIENT_ID        = Math.random().toString(36).slice(2);
const CONTROLS_TIMEOUT = 3000;
const SEND_EVERY_N     = 15; // send accumulated frames every N frames (~0.5s at 30fps)

// ── Helpers ───────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function getApiBase() {
  return $('backendUrl')?.value?.trim().replace(/\/$/, '') || 'http://localhost:8000';
}
function getWsBase() {
  return getApiBase().replace(/^https/, 'wss').replace(/^http/, 'ws');
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

let celebCtx          = null;
let ws                = null;
let holistic          = null;
let holisticCamera    = null;
let isRunning         = false;
let lastSentence      = '';
let sentenceTotal     = 0;
let handVisible       = false;
let lastSignText      = '';
let lastChipSigns     = [];
let controlsHideTimer = null;
let frameCount        = 0;

// ── Load MediaPipe scripts dynamically ────────────────────────────────────────
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
}

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
btnStart?.addEventListener('click', e => addRipple(btnStart, e));

// ── Controls auto-hide ────────────────────────────────────────────────────────
function showControls() {
  controls?.classList.remove('hidden');
  topBar?.classList.remove('hidden');
  clearTimeout(controlsHideTimer);
  controlsHideTimer = setTimeout(() => {
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
    const r = await fetch(`${getApiBase()}/status`, { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    if (d.model_loaded) {
      setStatus('ok', `${d.num_signs} signs`);
      const info = `✓ ${d.num_signs} signs loaded\nModel: ${d.model || 'hoyso48'}\n${d.signs.slice(0,10).join(', ')}…`;
      if (modelInfo) modelInfo.textContent = info;
    } else {
      setStatus('warn', 'No model');
      if (modelInfo) modelInfo.textContent = '⚠ No model\nCopy model files to backend/model/';
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
  if (ws && ws.readyState <= 1) return;
  ws = new WebSocket(`${getWsBase()}/ws/${CLIENT_ID}`);
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'prediction') {
      updateHUD(msg.sign, msg.confidence);
      updateChips(msg.buffer || []);
    }
    if (msg.type === 'sentence') {
      addSentence(msg.sentence, msg.gloss);
      updateChips([]);
    }
  };
  ws.onclose = () => setTimeout(connectWS, 3000);
}

// ── MediaPipe Holistic ────────────────────────────────────────────────────────
async function initHolistic() {
  toast('Loading hand detection…');

  // Load all three MediaPipe scripts
  await loadScript(HOLISTIC_UTILS);
  await loadScript(DRAWING_UTILS);
  await loadScript(HOLISTIC_SCRIPT);

  holistic = new window.Holistic({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`
  });

  holistic.setOptions({
    modelComplexity:           1,
    smoothLandmarks:           true,
    enableSegmentation:        false,
    smoothSegmentation:        false,
    refineFaceLandmarks:       false,
    minDetectionConfidence:    0.5,
    minTrackingConfidence:     0.5,
  });

  holistic.onResults(onHolisticResults);

  toast('Ready — show your hands ✋');
}

// ── Draw landmarks on canvas ──────────────────────────────────────────────────
function onHolisticResults(results) {
  if (!canvas || !ctx) return;

  // Resize canvas to match video
  if (canvas.width  !== video.videoWidth)  canvas.width  = video.videoWidth;
  if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Detect hand presence for pulse ring
  const hasHand = !!(results.leftHandLandmarks || results.rightHandLandmarks);
  if (hasHand !== handVisible) {
    handVisible = hasHand;
    handRing?.classList.toggle('active', hasHand);
  }

  // Draw hand connections
  if (results.leftHandLandmarks) {
    window.drawConnectors(ctx, results.leftHandLandmarks,
      window.HAND_CONNECTIONS, { color: 'rgba(41,196,154,.35)', lineWidth: 1.5 });
    window.drawLandmarks(ctx, results.leftHandLandmarks,
      { color: 'rgba(41,196,154,.8)', lineWidth: 1, radius: 3 });
  }
  if (results.rightHandLandmarks) {
    window.drawConnectors(ctx, results.rightHandLandmarks,
      window.HAND_CONNECTIONS, { color: 'rgba(41,196,154,.35)', lineWidth: 1.5 });
    window.drawLandmarks(ctx, results.rightHandLandmarks,
      { color: 'rgba(41,196,154,.8)', lineWidth: 1, radius: 3 });
  }

  // Send frame data to backend
  if (ws?.readyState === 1) {
    frameCount++;

    // Send each frame as it arrives
    ws.send(JSON.stringify({
      action: 'frame',
      landmarks: {
        faceLandmarks:      results.faceLandmarks      || null,
        poseLandmarks:      results.poseLandmarks      || null,
        leftHandLandmarks:  results.leftHandLandmarks  || null,
        rightHandLandmarks: results.rightHandLandmarks || null,
      }
    }));

    // Every SEND_EVERY_N frames, trigger prediction
    if (frameCount % SEND_EVERY_N === 0) {
      ws.send(JSON.stringify({ action: 'predict' }));
      frameCount = 0;
    }
  }
}


// ── Start camera ──────────────────────────────────────────────────────────────
async function startCamera() {
  btnStart.disabled = true;
  btnStart.querySelector('.btn-label').textContent = 'Starting…';

  try {
    if (!holistic) await initHolistic();

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width:     { ideal: 640 },
        height:    { ideal: 480 },
        frameRate: { ideal: 15, max: 20 }
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    // Wait for video to have valid frames (not just readyState)
    const waitForValidFrame = (timeout = 3000) => new Promise((resolve, reject) => {
      const start = Date.now();
      const check = () => {
        if (video.videoWidth > 0 && video.videoHeight > 0 && video.currentTime > 0) {
          // Optional: grab a frame to confirm it's not green (if needed)
          resolve();
        } else if (Date.now() - start > timeout) {
          reject(new Error('Video never produced valid frames'));
        } else {
          requestAnimationFrame(check);
        }
      };
      check();
    });

    await waitForValidFrame(4000); // wait up to 4 seconds for DroidCam

    // Optional: verify by drawing a frame to a temporary canvas (not needed usually)
    // const testCanvas = document.createElement('canvas');
    // testCanvas.width = video.videoWidth;
    // testCanvas.height = video.videoHeight;
    // testCanvas.getContext('2d').drawImage(video, 0, 0);
    // // check pixel data for green (if necessary)

    holisticCamera = new window.Camera(video, {
      onFrame: async () => {
        // Skip frames that are not ready
        if (!video.videoWidth || !video.videoHeight) return;
        if (video.readyState < 2) return;
        // Additional check: ensure timestamp is valid (avoid NaN issues)
        if (isNaN(video.currentTime)) return;
        await holistic.send({ image: video });
      },
      width:  640,
      height: 480,
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
    console.error('Camera init error:', err);
    btnStart.disabled = false;
    btnStart.querySelector('.btn-label').textContent = 'Start Camera';
    toast(`✕ Camera error: ${err.message}`, 5000);
  }
}

// ── HUD ───────────────────────────────────────────────────────────────────────
function updateHUD(sign, conf) {
  const thresh  = parseInt(confThresh?.value ?? 50) / 100;
  const valid   = !!(sign && conf >= thresh);
  const newSign = valid && sign !== lastSignText;

  if (newSign)  lastSignText = sign;
  if (!valid)   lastSignText = '';

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
  if (sentenceChips) sentenceChips.innerHTML = signs.map((s, i) =>
    `<span class="s-chip${i === signs.length-1 ? ' new' : ''}">${s}</span>`
  ).join('');
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
  if (!celebCtx) {
    if (!celebCanvas) return;
    celebCtx = celebCanvas.getContext('2d');
  }
  if (celebCanvas.width !== window.innerWidth)   celebCanvas.width  = window.innerWidth;
  if (celebCanvas.height !== window.innerHeight) celebCanvas.height = window.innerHeight;
  celebCanvas.style.display = 'block';

  const colors = ['#29c49a','#4dd9b4','#1a9e7c','#a8f0dc','#e8e8ec'];
  const pieces = Array.from({ length: 60 }, () => ({
    x: Math.random()*celebCanvas.width, y: -10,
    vx: (Math.random()-.5)*3.5, vy: Math.random()*4+2,
    rot: Math.random()*360, vr: (Math.random()-.5)*6,
    w: Math.random()*6+3, h: Math.random()*3+2,
    c: colors[Math.floor(Math.random()*colors.length)], a: 1,
  }));

  let frame = 0;
  (function draw() {
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
  ws?.readyState === 1 && ws.send(JSON.stringify({ action: 'flush' }));
  toast('Sentence flushed');
}

// ── Panels ────────────────────────────────────────────────────────────────────
btnSettings?.addEventListener('click', () => {
  settingsPanel?.classList.add('open');
  clearTimeout(controlsHideTimer);
});
btnSettingsClose?.addEventListener('click', () => {
  settingsPanel?.classList.remove('open');
  showControls();
});
settingsPanel?.addEventListener('click', (e) => {
  if (e.target === settingsPanel) settingsPanel.classList.remove('open');
});
sentenceText?.addEventListener('click', () => {
  if (sentenceTotal > 0) historyPanel?.classList.add('open');
});
btnHistoryClose?.addEventListener('click', () => historyPanel?.classList.remove('open'));
historyPanel?.addEventListener('click', (e) => {
  if (e.target === historyPanel) historyPanel.classList.remove('open');
});

// ── Sliders ───────────────────────────────────────────────────────────────────
confThresh?.addEventListener('input', () => {
  if (confThreshVal) confThreshVal.textContent = `${confThresh.value}%`;
});
pauseThresh?.addEventListener('input', () => {
  const val = (parseInt(pauseThresh.value)/10).toFixed(1);
  if (pauseThreshVal) pauseThreshVal.textContent = `${val}s`;
  ws?.readyState === 1 && ws.send(JSON.stringify({ action: 'set_pause', value: parseFloat(val) }));
});

// ── Button wiring ─────────────────────────────────────────────────────────────
btnStart?.addEventListener('click', startCamera);
btnFlush?.addEventListener('click', flush);
btnClear?.addEventListener('click', clearAll);
btnSpeak?.addEventListener('click', () => { if (lastSentence) speak(lastSentence); });

// ── Boot ──────────────────────────────────────────────────────────────────────
if (sentenceText) sentenceText.classList.add('placeholder');
checkBackend();
window.speechSynthesis?.getVoices();
speechSynthesis.addEventListener?.('voiceschanged', () => speechSynthesis.getVoices());
console.log('%c✋ ASL Translator', 'font-size:18px;font-weight:bold;color:#29c49a');
console.log('%cPowered by hoyso48 TFLite model (Kaggle 1st place)', 'color:#7a7a8e');
