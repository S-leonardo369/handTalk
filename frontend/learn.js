/**
 * handTalk Learn — learn.js
 * Library + Practice + Quiz using the existing FastAPI WebSocket inference engine.
 * No changes to main.py, app.js, or index.html required.
 */

// ── Config ────────────────────────────────────────────────────────────────────
const API_BASE       = localStorage.getItem('ht_backend') || 'http://localhost:8000';
const WS_BASE        = API_BASE.replace(/^https/, 'wss').replace(/^http/, 'ws');
const LEARN_ID       = 'learn_' + Math.random().toString(36).slice(2);
const REQUIRED_HITS  = 3;      // consecutive correct predictions to pass
const QUIZ_TIME      = 10;     // seconds per quiz sign
const INSTANT_CONF   = 0.85;   // confidence threshold for instant pass
const SEND_EVERY_N   = 15;     // frames per predict cycle (matches Translate)

const HOLISTIC_UTILS  = 'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3.1675466862/camera_utils.js';
const HOLISTIC_DRAW   = 'https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1675466124/drawing_utils.js';
const HOLISTIC_SCRIPT = 'https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/holistic.js';

const DRAW_CONN = Object.freeze({ color: 'rgba(41,196,154,.35)', lineWidth: 1.5 });
const DRAW_LM   = Object.freeze({ color: 'rgba(41,196,154,.8)',  lineWidth: 1, radius: 3 });

// ── Categories (static mapping — no backend change needed) ────────────────────
const CATEGORIES = {
  family:   ['mom','dad','brother','sister','baby','family','grandma','grandpa','uncle','boy','girl','man','woman'],
  feelings: ['happy','sad','angry','scared','tired','love','hate','sorry','proud','funny','boring','hurt','fine','upset','mad','cry','enjoy','worry'],
  actions:  ['eat','drink','sleep','play','work','help','go','come','want','need','learn','read','write','swim','run','walk','drive','sit','stand','look','listen','talk','hear','make','give','take','put','get','find','buy','pay','call','clean','cook','cut','draw','fall','finish','fix','forget','hide','hold','kiss','know','meet','open','pull','ride','run','say','search','see','send','share','show','sign','start','stay','stop','study','swim','take','taste','teach','tell','think','throw','turn','use','wait','wake','wash','watch','win','write'],
  objects:  ['book','phone','car','house','food','water','milk','ball','dog','cat','bird','fish','clock','chair','table','bed','bike','boat','box','bus','cake','candy','chicken','clothes','coat','computer','cup','door','duck','egg','fan','flag','flower','game','hat','horse','key','lion','medicine','money','moon','picture','pizza','rain','shoe','sky','snow','socks','star','sun','toy','tree','truck'],
  greetings:['hello','thank-you','nice','meet','good','morning','afternoon','night','please','sorry','yes','no'],
  time:     ['today','tomorrow','yesterday','now','later','morning','night','week','year','time','day','month','again','after','before','already','always','never','soon','still','when'],
  other:    [],
};

function getCategory(signName) {
  const s = signName.toLowerCase().replace(/-/g,' ');
  for (const [cat, list] of Object.entries(CATEGORIES)) {
    if (cat === 'other') continue;
    if (list.includes(s) || list.includes(signName.toLowerCase())) return cat;
  }
  return 'other';
}

// ── Progress (localStorage) ───────────────────────────────────────────────────
const PROGRESS_KEY = 'handTalk_progress';

function loadProgress() {
  try {
    const raw = localStorage.getItem(PROGRESS_KEY);
    if (raw) return JSON.parse(raw);
  } catch {}
  return { version: 1, mastered: [], weak: [], attempts: {}, bestTimes: {}, streak: 0, lastDate: null, quizHistory: [] };
}

function saveProgress(p) {
  try { localStorage.setItem(PROGRESS_KEY, JSON.stringify(p)); } catch {}
}

function updateStreak(p) {
  const today = new Date().toISOString().slice(0, 10);
  if (p.lastDate === today) return;
  const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
  p.streak    = (p.lastDate === yesterday) ? (p.streak + 1) : 1;
  p.lastDate  = today;
}

function markMastered(p, sign) {
  p.attempts[sign] = (p.attempts[sign] || 0) + 1;
  if (!p.mastered.includes(sign)) p.mastered.push(sign);
  p.weak = p.weak.filter(s => s !== sign);
}

function markWeak(p, sign) {
  if (!p.weak.includes(sign)) p.weak.push(sign);
}

let progress = loadProgress();

// ── Vocabulary ────────────────────────────────────────────────────────────────
let VOCAB = [];  // [{sign_id, sign, yt_embedId}]

async function fetchVocab() {
  try {
    const r = await fetch(`${API_BASE}/vocab`, { signal: AbortSignal.timeout(6000) });
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();
    VOCAB = (d.signs || []).sort((a, b) => a.sign.localeCompare(b.sign));
    toast(`${VOCAB.length} signs loaded`, 'success');
  } catch (e) {
    toast('⚠ Could not load sign library — is the backend running?', 'danger', 5000);
  }
  renderLibrary();
  renderPickerGrid();
}

function enriched(entry) {
  return {
    ...entry,
    mastered:  progress.mastered.includes(entry.sign),
    weak:      progress.weak.includes(entry.sign),
    attempts:  progress.attempts[entry.sign] || 0,
    category:  getCategory(entry.sign),
  };
}

// ── Toast ─────────────────────────────────────────────────────────────────────
const toastEl = document.getElementById('lrn-toast');
let toastTimer;
function toast(msg, type = '', ms = 2400) {
  toastEl.textContent = msg;
  toastEl.className   = `show${type ? ' ' + type : ''}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toastEl.className = ''; }, ms);
}

// ── Mode tab switching ────────────────────────────────────────────────────────
const tabs   = document.querySelectorAll('.mode-tab');
const panels = document.querySelectorAll('.mode-panel');

function switchTab(tabName) {
  tabs.forEach(t   => t.classList.toggle('active', t.dataset.tab === tabName));
  panels.forEach(p => p.classList.toggle('active', p.id === `panel-${tabName}`));
  if (tabName === 'library' || (tabName === 'practice' && !session.active)) {
    stopCamera();
  }
}

tabs.forEach(t => t.addEventListener('click', () => switchTab(t.dataset.tab)));

// ── LIBRARY ───────────────────────────────────────────────────────────────────
const signGrid  = document.getElementById('signGrid');
const libSearch = document.getElementById('libSearch');
const catFilter = document.getElementById('catFilter');
const libStats  = document.getElementById('libStats');

function renderLibrary() {
  const q   = libSearch.value.trim().toLowerCase();
  const cat = catFilter.value;

  let items = VOCAB.map(enriched);
  if (cat !== 'all') items = items.filter(e => e.category === cat);
  if (q)             items = items.filter(e => e.sign.toLowerCase().includes(q));

  libStats.textContent = `${items.length} signs`;

  if (!items.length) {
    signGrid.innerHTML = `<div class="empty-state" style="grid-column:1/-1">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>
      <p>No signs match "${q}"</p></div>`;
    return;
  }

  signGrid.innerHTML = items.map(e => `
    <div class="sign-card${e.mastered ? ' mastered' : e.weak ? ' weak' : ''}"
         data-sign="${e.sign}" data-yt="${e.yt_embedId || ''}">
      <span class="sign-card-badge"></span>
      <div class="sign-card-thumb">${e.sign.slice(0,2).toUpperCase()}</div>
      <div class="sign-card-name">${e.sign}</div>
    </div>`).join('');

  signGrid.querySelectorAll('.sign-card').forEach(card => {
    card.addEventListener('click', () => openSignModal(card.dataset.sign, card.dataset.yt));
  });
}

libSearch.addEventListener('input', renderLibrary);
catFilter.addEventListener('change', renderLibrary);

// ── Sign modal ────────────────────────────────────────────────────────────────
const modal         = document.getElementById('signModal');
const modalSignName = document.getElementById('modalSignName');
const modalVideoSlot= document.getElementById('modalVideoSlot');
const modalNoVideo  = document.getElementById('modalNoVideo');
let   modalCurrentSign = null;

function openSignModal(sign, ytId) {
  modalCurrentSign    = sign;
  modalSignName.textContent = sign.toUpperCase();
  modal.classList.remove('hidden');

  if (ytId) {
    modalNoVideo.classList.add('hidden');
    modalVideoSlot.innerHTML = `<iframe
      style="position:absolute;inset:0;width:100%;height:100%;border:none"
      src="https://www.youtube.com/embed/${ytId}?autoplay=1&mute=1&rel=0"
      allow="autoplay; encrypted-media" allowfullscreen></iframe>`;
  } else {
    modalNoVideo.classList.remove('hidden');
    modalVideoSlot.innerHTML = '';
  }
}

function closeModal() {
  modal.classList.add('hidden');
  modalVideoSlot.innerHTML = '';  // stop video
}

document.getElementById('modalClose').addEventListener('click', closeModal);
document.getElementById('modalClose2').addEventListener('click', closeModal);
modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });

document.getElementById('modalPractice').addEventListener('click', () => {
  if (!modalCurrentSign) return;
  closeModal();
  startSinglePractice(modalCurrentSign);
});

// ── Practice picker grid ──────────────────────────────────────────────────────
const practicePickerGrid = document.getElementById('practicePickerGrid');

function renderPickerGrid() {
  const items = VOCAB.map(enriched);
  practicePickerGrid.innerHTML = items.map(e => `
    <div class="sign-card${e.mastered ? ' mastered' : e.weak ? ' weak' : ''}"
         data-sign="${e.sign}">
      <span class="sign-card-badge"></span>
      <div class="sign-card-thumb">${e.sign.slice(0,2).toUpperCase()}</div>
      <div class="sign-card-name">${e.sign}</div>
    </div>`).join('');

  practicePickerGrid.querySelectorAll('.sign-card').forEach(card => {
    card.addEventListener('click', () => startSinglePractice(card.dataset.sign));
  });

  const weakBtn = document.getElementById('btnStartWeak');
  if (progress.weak.length > 0) weakBtn.style.display = '';
  else weakBtn.style.display = 'none';
}

// ── MediaPipe / Camera helpers ────────────────────────────────────────────────
let holisticLoaded = false;
let holisticInst   = null;

async function loadHolistic() {
  if (holisticLoaded) return;
  await loadScript(HOLISTIC_UTILS);
  await loadScript(HOLISTIC_DRAW);
  await loadScript(HOLISTIC_SCRIPT);
  holisticLoaded = true;
}

function loadScript(src) {
  return new Promise((res, rej) => {
    if (document.querySelector(`script[src="${src}"]`)) { res(); return; }
    const s = document.createElement('script');
    s.src = src; s.onload = res; s.onerror = rej;
    document.head.appendChild(s);
  });
}

async function startCamera(videoEl, canvasEl, placeholderEl, onResults) {
  await loadHolistic();

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (e) {
    toast('✕ Camera access denied', 'danger', 4000);
    return null;
  }

  videoEl.srcObject = stream;
  await videoEl.play();
  videoEl.style.display = '';
  canvasEl.style.display = '';
  if (placeholderEl) placeholderEl.style.display = 'none';

  const h = new window.Holistic({
    locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${f}`
  });
  h.setOptions({
    modelComplexity: 1, smoothLandmarks: true,
    enableSegmentation: false, smoothSegmentation: false,
    refineFaceLandmarks: false,
    minDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  h.onResults(onResults);

  const cam = new window.Camera(videoEl, {
    onFrame: async () => { if (h) await h.send({ image: videoEl }); },
    width: 1280, height: 720,
  });
  cam.start();

  return { stream, holistic: h, camera: cam };
}

function stopCameraHandle(handle) {
  if (!handle) return;
  try { handle.camera.stop(); } catch {}
  try { handle.holistic.close(); } catch {}
  handle.stream.getTracks().forEach(t => t.stop());
}

function packLandmarks(res) {
  const o = {};
  if (res.faceLandmarks)      o.faceLandmarks      = res.faceLandmarks;
  if (res.poseLandmarks)      o.poseLandmarks       = res.poseLandmarks;
  if (res.leftHandLandmarks)  o.leftHandLandmarks   = res.leftHandLandmarks;
  if (res.rightHandLandmarks) o.rightHandLandmarks  = res.rightHandLandmarks;
  return o;
}

function drawOnCanvas(results, videoEl, canvasEl) {
  canvasEl.width  = videoEl.videoWidth  || 640;
  canvasEl.height = videoEl.videoHeight || 480;
  const ctx = canvasEl.getContext('2d');
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  if (results.leftHandLandmarks) {
    window.drawConnectors(ctx, results.leftHandLandmarks, window.HAND_CONNECTIONS, DRAW_CONN);
    window.drawLandmarks(ctx,  results.leftHandLandmarks, DRAW_LM);
  }
  if (results.rightHandLandmarks) {
    window.drawConnectors(ctx, results.rightHandLandmarks, window.HAND_CONNECTIONS, DRAW_CONN);
    window.drawLandmarks(ctx,  results.rightHandLandmarks, DRAW_LM);
  }
}

// ── WebSocket helpers ─────────────────────────────────────────────────────────
let ws           = null;
let wsFrameCount = 0;
let wsPrevXY     = null;
const PREDICT_MSG = '{"action":"predict"}';

function connectLearnWS(onMsg) {
  if (ws && ws.readyState <= WebSocket.OPEN) return;
  ws = new WebSocket(`${WS_BASE}/ws/${LEARN_ID}`);
  ws.onopen = () => {
    // Tighter gates for scoring — fewer false positives
    ws.send(JSON.stringify({
      action: 'set_threshold',
      confidence: 0.45,
      margin:     0.10,
      consecutive: 1,   // backend consecutive is irrelevant — we handle it in JS
    }));
  };
  ws.onmessage = (e) => {
    let msg; try { msg = JSON.parse(e.data); } catch { return; }
    onMsg(msg);
  };
  ws.onerror  = () => {};
  ws.onclose  = () => {};
}

function disconnectLearnWS() {
  if (ws) { try { ws.close(); } catch {} ws = null; }
  wsFrameCount = 0; wsPrevXY = null;
}

function sendFrame(landmarks) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const lh = landmarks.leftHandLandmarks  || [];
  const rh = landmarks.rightHandLandmarks || [];
  if (!lh.length && !rh.length) return;

  // Motion gate — skip static frames
  const pts = lh.concat(rh);
  const xy  = pts.map(p => [p.x, p.y]);
  if (wsPrevXY && wsPrevXY.length === xy.length) {
    let delta = 0;
    for (let i = 0; i < xy.length; i++) delta += Math.abs(xy[i][0] - wsPrevXY[i][0]) + Math.abs(xy[i][1] - wsPrevXY[i][1]);
    if (delta / xy.length < 0.012) { wsPrevXY = xy; return; }
  }
  wsPrevXY = xy;

  ws.send(JSON.stringify({ action: 'frame', landmarks }));
  wsFrameCount++;
  if (wsFrameCount >= SEND_EVERY_N) {
    ws.send(PREDICT_MSG);
    wsFrameCount = 0;
  }
}

// ── Ring helper ───────────────────────────────────────────────────────────────
function setRing(ringEl, fraction, state) {
  // fraction 0..1; state: '' | 'wrong' | 'done' | 'danger'
  const circ   = 2 * Math.PI * 35;
  const offset = circ * (1 - Math.max(0, Math.min(1, fraction)));
  ringEl.style.strokeDashoffset = offset;
  ringEl.className = `ring-fill${state ? ' ' + state : ''}`;
}

// ── SESSION STATE ─────────────────────────────────────────────────────────────
const session = {
  active:       false,
  queue:        [],
  queueIndex:   0,
  currentTarget:null,
  hitStreak:    0,
  signStartTime:null,
  results:      [],
  cameraHandle: null,
};

function stopCamera() {
  stopCameraHandle(session.cameraHandle);
  session.cameraHandle = null;
  disconnectLearnWS();
}

// ── PRACTICE SESSION ──────────────────────────────────────────────────────────
const signPicker       = document.getElementById('signPicker');
const practiceSession  = document.getElementById('practiceSession');
const sessionTargetWord= document.getElementById('sessionTargetWord');
const sessionProgress  = document.getElementById('sessionProgress');
const sessionProgFill  = document.getElementById('sessionProgFill');
const practiceRing     = document.getElementById('practiceRing');
const practiceConfFill = document.getElementById('practiceConfFill');
const practiceConfVal  = document.getElementById('practiceConfVal');
const practiceCamStatus= document.getElementById('practiceCamStatus');
const arenaVideoSlot   = document.getElementById('arenaVideoSlot');
const sessionComplete  = document.getElementById('sessionComplete');
const completeResults  = document.getElementById('completeResults');

function buildQueue(signs) {
  // Shuffle, but front-load weak signs
  const weak   = signs.filter(s => progress.weak.includes(s));
  const normal = signs.filter(s => !progress.weak.includes(s));
  const shuffle = arr => arr.sort(() => Math.random() - .5);
  return [...shuffle(weak), ...shuffle(normal)];
}

function startSession(signNames) {
  session.queue      = buildQueue(signNames);
  session.queueIndex = 0;
  session.results    = [];
  session.active     = true;

  signPicker.style.display      = 'none';
  practiceSession.classList.add('active');
  sessionComplete.classList.remove('active');

  loadPracticeSign(session.queue[0]);
  initPracticeCamera();
}

function startSinglePractice(sign) {
  switchTab('practice');
  const queue = [sign, ...VOCAB.map(e => e.sign).filter(s => s !== sign && !progress.mastered.includes(s)).slice(0, 9)];
  startSession(queue.slice(0, Math.min(queue.length, 10)));
}

function loadPracticeSign(sign) {
  session.currentTarget = sign;
  session.hitStreak     = 0;
  session.signStartTime = performance.now();

  sessionTargetWord.textContent = sign.toUpperCase();
  const idx = session.queueIndex + 1;
  sessionProgress.textContent   = `${idx} / ${session.queue.length}`;
  sessionProgFill.style.width   = `${((idx - 1) / session.queue.length) * 100}%`;
  practiceConfFill.style.width  = '0%';
  practiceConfVal.textContent   = '—';
  practiceCamStatus.textContent = 'Show your hands ✋';
  setRing(practiceRing, 0, '');
  document.getElementById('btnGotIt').disabled = true;

  // Load demo video
  const entry = VOCAB.find(e => e.sign === sign);
  if (entry?.yt_embedId) {
    arenaVideoSlot.innerHTML = `<iframe
      style="position:absolute;inset:0;width:100%;height:100%;border:none"
      src="https://www.youtube.com/embed/${entry.yt_embedId}?autoplay=1&mute=1&loop=1&rel=0"
      allow="autoplay; encrypted-media" allowfullscreen></iframe>`;
  } else {
    arenaVideoSlot.innerHTML = `<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--text-dim);font-size:13px">No demo video</div>`;
  }
}

async function initPracticeCamera() {
  const videoEl       = document.getElementById('practiceVideo');
  const canvasEl      = document.getElementById('practiceCanvas');
  const placeholderEl = document.getElementById('practiceCamPlaceholder');

  session.cameraHandle = await startCamera(videoEl, canvasEl, placeholderEl, onPracticeResults);
  if (!session.cameraHandle) return;

  connectLearnWS(onPracticeWS);
  practiceCamStatus.textContent = 'Ready — sign when you are';
}

function onPracticeResults(results) {
  const videoEl  = document.getElementById('practiceVideo');
  const canvasEl = document.getElementById('practiceCanvas');
  drawOnCanvas(results, videoEl, canvasEl);
  sendFrame(packLandmarks(results));
}

function onPracticeWS(msg) {
  if (msg.type !== 'prediction') return;
  const sign  = msg.sign;
  const conf  = msg.confidence || 0;
  const isTarget = sign === session.currentTarget;

  // Update confidence bar
  const pct  = Math.round(conf * 100);
  practiceConfFill.style.width = `${pct}%`;
  practiceConfFill.className   = `conf-bar-fill${isTarget ? '' : ' wrong'}`;
  practiceConfVal.textContent  = `${pct}%`;

  if (sign) {
    practiceCamStatus.textContent = isTarget
      ? `✓ ${sign} detected`
      : `Model sees: ${sign}`;
  } else {
    practiceCamStatus.textContent = msg.gate ? `(${msg.gate.replace(/_/g,' ')})` : 'Listening…';
  }

  if (isTarget) {
    session.hitStreak++;
    const fraction = session.hitStreak / REQUIRED_HITS;
    setRing(practiceRing, fraction, fraction >= 1 ? 'done' : '');

    if (conf >= INSTANT_CONF || session.hitStreak >= REQUIRED_HITS) {
      passCurrentSign();
    }
  } else {
    if (session.hitStreak > 0) session.hitStreak = Math.max(0, session.hitStreak - 1);
    setRing(practiceRing, session.hitStreak / REQUIRED_HITS, sign ? 'wrong' : '');
  }
}

function passCurrentSign() {
  const elapsed = ((performance.now() - session.signStartTime) / 1000).toFixed(1);
  progress = loadProgress();
  markMastered(progress, session.currentTarget);
  if (elapsed < (progress.bestTimes[session.currentTarget] || 999)) {
    progress.bestTimes[session.currentTarget] = parseFloat(elapsed);
  }
  updateStreak(progress);
  saveProgress(progress);

  session.results.push({ sign: session.currentTarget, passed: true, time: elapsed, skipped: false });
  toast(`✓ ${session.currentTarget.toUpperCase()} — ${elapsed}s`, 'success', 1600);

  setRing(practiceRing, 1, 'done');
  setTimeout(advanceSession, 900);
}

function advanceSession() {
  session.queueIndex++;
  if (session.queueIndex >= session.queue.length) {
    endSession();
    return;
  }
  loadPracticeSign(session.queue[session.queueIndex]);
}

function endSession() {
  sessionProgFill.style.width = '100%';
  sessionComplete.classList.add('active');
  stopCamera();
  session.active = false;

  // Update weak signs from failures
  progress = loadProgress();
  session.results.forEach(r => { if (!r.passed) markWeak(progress, r.sign); });
  saveProgress(progress);
  renderPickerGrid();

  // Render results
  const passed = session.results.filter(r => r.passed).length;
  completeResults.innerHTML = `
    <div style="font-size:18px;font-weight:800;margin-bottom:8px">
      ${passed} / ${session.results.length} passed
    </div>` +
    session.results.map(r => `
      <div class="result-row ${r.passed ? 'pass' : 'fail'}">
        <span class="result-icon">${r.passed ? '✓' : '✗'}</span>
        <span class="result-sign">${r.sign.toUpperCase()}</span>
        <span class="result-time">${r.skipped ? 'skipped' : r.passed ? `${r.time}s` : 'not signed'}</span>
      </div>`).join('');
}

// Practice buttons
document.getElementById('btnStartSession').addEventListener('click', () => {
  const signs = VOCAB.map(e => e.sign).filter(s => !progress.mastered.includes(s));
  const pool  = signs.length >= 10 ? signs : VOCAB.map(e => e.sign);
  startSession(buildQueue(pool).slice(0, 10));
});

document.getElementById('btnStartWeak').addEventListener('click', () => {
  if (!progress.weak.length) { toast('No weak signs yet — complete a quiz first!'); return; }
  startSession(progress.weak.slice());
});

document.getElementById('btnSkipSign').addEventListener('click', () => {
  session.results.push({ sign: session.currentTarget, passed: false, time: null, skipped: true });
  advanceSession();
});

document.getElementById('btnGotIt').addEventListener('click', passCurrentSign);

document.getElementById('btnSessionBack').addEventListener('click', () => {
  stopCamera();
  session.active = false;
  practiceSession.classList.remove('active');
  signPicker.style.display = '';
});

document.getElementById('btnPracticeAgain').addEventListener('click', () => {
  sessionComplete.classList.remove('active');
  const weak = session.results.filter(r => !r.passed).map(r => r.sign);
  if (weak.length) startSession(weak);
  else { practiceSession.classList.remove('active'); signPicker.style.display = ''; }
});

document.getElementById('btnGoQuiz').addEventListener('click', () => {
  stopCamera(); session.active = false;
  practiceSession.classList.remove('active');
  signPicker.style.display = '';
  switchTab('quiz');
});

document.getElementById('btnBackLibrary').addEventListener('click', () => {
  stopCamera(); session.active = false;
  practiceSession.classList.remove('active');
  signPicker.style.display = '';
  switchTab('library');
});

// ── QUIZ ──────────────────────────────────────────────────────────────────────
const quizStart        = document.getElementById('quizStart');
const quizActive       = document.getElementById('quizActive');
const quizResults      = document.getElementById('quizResults');
const quizProgressText = document.getElementById('quizProgressText');
const quizTimerDisplay = document.getElementById('quizTimerDisplay');
const timerBarFill     = document.getElementById('timerBarFill');
const quizTargetWord   = document.getElementById('quizTargetWord');
const quizModelSees    = document.getElementById('quizModelSees');
const scoreRingFill    = document.getElementById('scoreRingFill');
const scoreNum         = document.getElementById('scoreNum');
const scoreDen         = document.getElementById('scoreDen');
const resultsTable     = document.getElementById('resultsTable');

const quiz = {
  queue:        [],
  queueIndex:   0,
  currentTarget:null,
  hitStreak:    0,
  signStartTime:null,
  results:      [],
  timerSecs:    QUIZ_TIME,
  timerInterval:null,
  cameraHandle: null,
};

let quizCount = 10;
document.querySelectorAll('.quiz-opt').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.quiz-opt').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    quizCount = parseInt(btn.dataset.count);
  });
});

document.getElementById('btnStartQuiz').addEventListener('click', startQuiz);

async function startQuiz() {
  quizStart.style.display  = 'none';
  quizActive.classList.add('running');
  quizResults.classList.remove('visible');

  const pool = VOCAB.map(e => e.sign);
  quiz.queue        = buildQueue(pool).slice(0, quizCount);
  quiz.queueIndex   = 0;
  quiz.results      = [];
  quiz.cameraHandle = null;

  // Init camera
  const videoEl  = document.getElementById('quizVideo');
  const canvasEl = document.getElementById('quizCanvas');
  const placeholder = document.getElementById('quizCamPlaceholder');
  quiz.cameraHandle = await startCamera(videoEl, canvasEl, placeholder, onQuizResults);
  connectLearnWS(onQuizWS);

  loadQuizSign(quiz.queue[0]);
}

function loadQuizSign(sign) {
  quiz.currentTarget = sign;
  quiz.hitStreak     = 0;
  quiz.signStartTime = performance.now();
  quiz.timerSecs     = QUIZ_TIME;

  quizTargetWord.textContent   = sign.toUpperCase();
  quizProgressText.textContent = `${quiz.queueIndex + 1} / ${quiz.queue.length}`;
  quizModelSees.innerHTML      = 'Model sees: <strong>—</strong>';
  timerBarFill.style.width     = '100%';
  timerBarFill.className       = 'timer-bar-fill';
  quizTimerDisplay.textContent = QUIZ_TIME;
  quizTimerDisplay.className   = 'quiz-timer-display';

  clearInterval(quiz.timerInterval);
  quiz.timerInterval = setInterval(tickQuizTimer, 1000);
}

function tickQuizTimer() {
  quiz.timerSecs--;
  quizTimerDisplay.textContent = quiz.timerSecs;
  timerBarFill.style.width     = `${(quiz.timerSecs / QUIZ_TIME) * 100}%`;

  if (quiz.timerSecs <= 3) {
    quizTimerDisplay.className = 'quiz-timer-display urgent';
    timerBarFill.className     = 'timer-bar-fill urgent';
  }

  if (quiz.timerSecs <= 0) {
    clearInterval(quiz.timerInterval);
    failQuizSign('timeout');
  }
}

function onQuizResults(results) {
  const videoEl  = document.getElementById('quizVideo');
  const canvasEl = document.getElementById('quizCanvas');
  drawOnCanvas(results, videoEl, canvasEl);
  sendFrame(packLandmarks(results));
}

function onQuizWS(msg) {
  if (msg.type !== 'prediction') return;
  const sign = msg.sign;
  const conf = msg.confidence || 0;

  if (sign) {
    quizModelSees.innerHTML = `Model sees: <strong>${sign} (${Math.round(conf*100)}%)</strong>`;
  } else {
    quizModelSees.innerHTML = `Model sees: <strong>—</strong>`;
  }

  const isTarget = sign === quiz.currentTarget;
  if (isTarget) {
    quiz.hitStreak++;
    if (conf >= INSTANT_CONF || quiz.hitStreak >= REQUIRED_HITS) {
      clearInterval(quiz.timerInterval);
      passQuizSign();
    }
  } else {
    if (quiz.hitStreak > 0) quiz.hitStreak = Math.max(0, quiz.hitStreak - 1);
  }
}

function passQuizSign() {
  const elapsed = ((performance.now() - quiz.signStartTime) / 1000).toFixed(1);
  quiz.results.push({ sign: quiz.currentTarget, passed: true, time: parseFloat(elapsed), reason: `${elapsed}s` });

  progress = loadProgress();
  markMastered(progress, quiz.currentTarget);
  if (parseFloat(elapsed) < (progress.bestTimes[quiz.currentTarget] || 999)) {
    progress.bestTimes[quiz.currentTarget] = parseFloat(elapsed);
  }
  saveProgress(progress);

  toast(`✓ ${quiz.currentTarget.toUpperCase()} — ${elapsed}s`, 'success', 1200);
  advanceQuiz();
}

function failQuizSign(reason) {
  quiz.results.push({ sign: quiz.currentTarget, passed: false, time: null, reason });
  progress = loadProgress();
  markWeak(progress, quiz.currentTarget);
  saveProgress(progress);
  advanceQuiz();
}

function advanceQuiz() {
  clearInterval(quiz.timerInterval);
  quiz.queueIndex++;
  if (quiz.queueIndex >= quiz.queue.length) {
    endQuiz();
    return;
  }
  setTimeout(() => loadQuizSign(quiz.queue[quiz.queueIndex]), 400);
}

function endQuiz() {
  stopCameraHandle(quiz.cameraHandle);
  quiz.cameraHandle = null;
  disconnectLearnWS();

  quizActive.classList.remove('running');
  quizResults.classList.add('visible');

  const passed = quiz.results.filter(r => r.passed).length;
  const total  = quiz.results.length;
  const best   = quiz.results.filter(r => r.passed).reduce((b, r) => (r.time < b ? r.time : b), 999);

  // Score ring animation
  const frac   = passed / total;
  const circ   = 2 * Math.PI * 45;
  scoreRingFill.style.strokeDashoffset = circ * (1 - frac);
  scoreNum.textContent = passed;
  scoreDen.textContent = `/ ${total}`;

  progress = loadProgress();
  updateStreak(progress);
  progress.quizHistory.push({ date: new Date().toISOString().slice(0,10), score: passed, total, duration: 0 });
  saveProgress(progress);

  document.getElementById('quizStreakDisplay').textContent = `🔥 ${progress.streak}`;
  document.getElementById('quizBestTime').textContent      = best < 999 ? `${best}s` : '—';

  resultsTable.innerHTML = quiz.results.map(r => `
    <div class="results-row ${r.passed ? 'pass' : 'fail'}">
      <span style="font-size:12px">${r.passed ? '✓' : '✗'}</span>
      <span class="results-sign">${r.sign}</span>
      <span class="results-time">${r.reason}</span>
      <span class="results-label">${r.passed ? 'PASSED' : 'FAILED'}</span>
    </div>`).join('');

  renderLibrary();
  renderPickerGrid();
}

document.getElementById('btnQuizSkip').addEventListener('click', () => {
  clearInterval(quiz.timerInterval);
  failQuizSign('skip');
});

document.getElementById('btnQuizAgain').addEventListener('click', () => {
  quizResults.classList.remove('visible');
  quizStart.style.display = '';
});

document.getElementById('btnQuizNew').addEventListener('click', () => {
  quizResults.classList.remove('visible');
  quizStart.style.display = '';
});

document.getElementById('btnQuizWeak').addEventListener('click', () => {
  quizResults.classList.remove('visible');
  quizStart.style.display = '';
  switchTab('practice');
  if (progress.weak.length) {
    startSession(progress.weak.slice());
  } else {
    toast('No weak signs — great job!', 'success');
  }
});

// ── Init ──────────────────────────────────────────────────────────────────────
fetchVocab();