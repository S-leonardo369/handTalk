/**
 * handTalk Sign — sign.js
 * Text / Voice → ASL sign video lookup.
 * Calls POST /text-to-sign on the FastAPI backend.
 * No WebSocket, no camera — pure lookup and display.
 */

// ── Config ────────────────────────────────────────────────────────────────────
const API_BASE = localStorage.getItem('ht_backend') || 'http://localhost:8000';
const MAX_CHARS = 300;
const PLAY_DELAY_MS = 2800;   // ms between cards in Play All mode

// ── DOM refs ──────────────────────────────────────────────────────────────────
const signInput      = document.getElementById('signInput');
const charCount      = document.getElementById('charCount');
const btnTranslate   = document.getElementById('btnTranslate');
const btnClear       = document.getElementById('btnClear');
const resultsSection = document.getElementById('resultsSection');
const skeletonSection= document.getElementById('skeletonSection');
const emptyState     = document.getElementById('emptyState');
const signCards      = document.getElementById('signCards');
const skeletonGrid   = document.getElementById('skeletonGrid');
const resultsTitle   = document.getElementById('resultsTitle');
const unknownBanner  = document.getElementById('unknownBanner');
const btnPlayAll     = document.getElementById('btnPlayAll');
const btnCopyWords   = document.getElementById('btnCopyWords');
const toastEl        = document.getElementById('sign-toast');

// ── State ─────────────────────────────────────────────────────────────────────
let currentResults  = [];   // [{word, sign, asl_vidref, found}]
let isPlayingAll    = false;
let playAllTimer    = null;
let recognition     = null;

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer;
function toast(msg, type = '', ms = 2400) {
  toastEl.textContent = msg;
  toastEl.className   = `show${type ? ' ' + type : ''}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toastEl.className = ''; }, ms);
}

// ── SignASL widget re-init (same as learn.js) ─────────────────────────────────
function reloadSignASLWidget() {
  const SRC = 'https://embed.signasl.org/widgets.js';
  const old = document.querySelector(`script[src="${SRC}"]`);
  if (old) old.remove();
  const s = document.createElement('script');
  s.src = SRC; s.charset = 'utf-8';
  document.body.appendChild(s);
}

// ── Text input ────────────────────────────────────────────────────────────────
signInput.addEventListener('input', () => {
  const len = signInput.value.length;
  charCount.textContent    = `${len} / ${MAX_CHARS}`;
  btnTranslate.disabled    = len === 0;
});

signInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!btnTranslate.disabled) translate();
  }
});

btnClear.addEventListener('click', () => {
  signInput.value          = '';
  charCount.textContent    = `0 / ${MAX_CHARS}`;
  btnTranslate.disabled    = true;
  clearResults();
  stopPlayAll();
  signInput.focus();
});

function clearResults() {
  currentResults = [];
  resultsSection.style.display  = 'none';
  skeletonSection.style.display = 'none';
  emptyState.style.display      = '';
  signCards.innerHTML            = '';
}


// ── Translate ─────────────────────────────────────────────────────────────────
btnTranslate.addEventListener('click', translate);

async function translate() {
  const text = signInput.value.trim();
  if (!text) return;

  stopPlayAll();
  showSkeleton(estimateWordCount(text));

  try {
    const res = await fetch(`${API_BASE}/text-to-sign`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text }),
      signal:  AbortSignal.timeout(8000),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    currentResults = data.results || [];
    renderResults(currentResults);

  } catch (e) {
    hideSkeleton();
    toast(`Translation failed — is the backend running?`, 'danger', 4000);
    console.error('[sign.js] translate error:', e);
  }
}

function estimateWordCount(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

// ── Skeleton loading ──────────────────────────────────────────────────────────
function showSkeleton(count) {
  emptyState.style.display      = 'none';
  resultsSection.style.display  = 'none';
  skeletonSection.style.display = '';
  skeletonGrid.innerHTML = Array.from({ length: Math.max(count, 3) }, () => `
    <div class="skeleton-card">
      <div class="skeleton-video"></div>
      <div class="skeleton-foot"></div>
    </div>`).join('');
}

function hideSkeleton() {
  skeletonSection.style.display = 'none';
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(results) {
  hideSkeleton();

  if (!results.length) {
    emptyState.style.display = '';
    return;
  }

  const found   = results.filter(r => r.found);
  const unknown = results.filter(r => !r.found);

  resultsTitle.textContent = `${found.length} sign${found.length !== 1 ? 's' : ''} found`;
  if (results.length !== found.length) {
    resultsTitle.textContent += ` · ${results.length} words`;
  }

  // Unknown words banner
  if (unknown.length) {
    unknownBanner.innerHTML = `<strong>${unknown.map(r => r.word).join(', ')}</strong>
      — not in the sign vocabulary (shown as text only)`;
    unknownBanner.classList.add('show');
  } else {
    unknownBanner.classList.remove('show');
  }

  signCards.innerHTML = results.map((r, i) => buildCard(r, i)).join('');


  resultsSection.style.display = '';
  emptyState.style.display     = 'none';

  btnPlayAll.disabled = found.length === 0;

  // Scroll to results
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// REPLACE THE WHOLE buildCard FUNCTION WITH THIS:
function buildCard(r, index) {
  const ytId = r.found && r.yt_embedId ? r.yt_embedId.trim() : '';

  const videoHTML = ytId
    ? `<div class="yt-thumb" data-ytid="${ytId}" style="position:relative;width:100%;height:100%;cursor:pointer;background:#000;">
         <img src="https://img.youtube.com/vi/${ytId}/mqdefault.jpg"
              style="width:100%;height:100%;object-fit:cover;" alt="ASL ${r.sign}" loading="lazy">
         <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none;">
           <svg width="48" height="48" viewBox="0 0 68 48">
             <path d="M66.5 7.7c-.8-2.9-3-5.2-5.9-6C55.8.5 34 .5 34 .5S12.2.5 7.4 1.7c-2.9.8-5.1 3.1-5.9 6C.3 12.5.3 24 .3 24s0 11.5 1.2 16.3c.8 2.9 3 5.2 5.9 6C12.2 47.5 34 47.5 34 47.5s21.8 0 26.6-1.2c2.9-.8 5.1-3.1 5.9-6 1.2-4.8 1.2-16.3 1.2-16.3s0-11.5-1.2-16.3z" fill="red"/>
             <path d="M27.1 34.6L44.9 24 27.1 13.4v21.2z" fill="white"/>
           </svg>
         </div>
       </div>`
    : `<div class="card-no-video">
         <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
           <path d="M23 7l-7 5 7 5V7z"/>
           <rect x="1" y="5" width="15" height="14" rx="2"/>
         </svg>
         <span>No sign found</span>
       </div>`;

  return `
    <div class="sign-result-card${r.found ? '' : ' unknown'}" data-index="${index}">
      <div class="card-video">${videoHTML}</div>
      <div class="card-playing-bar">
        <div class="card-playing-bar-fill" id="playbar-${index}"></div>
      </div>
      <div class="card-foot">
        <span class="card-word${r.found ? '' : ' unknown-word'}">${r.word}</span>
        <span class="card-index">${index + 1}</span>
      </div>
    </div>`;
}


// ── Play All ──────────────────────────────────────────────────────────────────
btnPlayAll.addEventListener('click', () => {
  if (isPlayingAll) { stopPlayAll(); return; }
  const found = currentResults.filter(r => r.found);
  if (!found.length) return;
  playAll(found);
});

function playAll(results) {
  isPlayingAll = true;
  btnPlayAll.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="4" width="4" height="16"/>
      <rect x="14" y="4" width="4" height="16"/>
    </svg> Stop`;

  let i = 0;

  function playNext() {
    if (!isPlayingAll || i >= results.length) {
      stopPlayAll(); return;
    }

    // Highlight current card
    document.querySelectorAll('.sign-result-card').forEach(c => c.classList.remove('playing'));
    const idx  = currentResults.indexOf(results[i]);
    const card = document.querySelector(`.sign-result-card[data-index="${idx}"]`);
    if (card) {
      card.classList.add('playing');
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

      // Animate progress bar
      const bar = document.getElementById(`playbar-${idx}`);
      if (bar) {
        bar.style.transition = 'none';
        bar.style.width      = '0%';
        requestAnimationFrame(() => {
          bar.style.transition = `width ${PLAY_DELAY_MS}ms linear`;
          bar.style.width      = '100%';
        });
      }
    }

    i++;
    playAllTimer = setTimeout(playNext, PLAY_DELAY_MS);
  }

  playNext();
}

function stopPlayAll() {
  isPlayingAll = false;
  clearTimeout(playAllTimer);
  document.querySelectorAll('.sign-result-card').forEach(c => c.classList.remove('playing'));
  document.querySelectorAll('.card-playing-bar-fill').forEach(b => {
    b.style.transition = 'none'; b.style.width = '0%';
  });
  btnPlayAll.innerHTML = `
    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5,3 19,12 5,21"/>
    </svg> Play All`;
}

// ── Copy words ────────────────────────────────────────────────────────────────
btnCopyWords.addEventListener('click', () => {
  const words = currentResults.map(r => r.word).join(' ');
  navigator.clipboard.writeText(words).then(() => {
    toast('Copied to clipboard', 'success');
  }).catch(() => toast('Copy failed'));
});

// Thumbnail click → load iframe (lazy load for speed)
signCards.addEventListener('click', (e) => {
  const thumb = e.target.closest('.yt-thumb');
  if (!thumb) return;
  const ytId = thumb.dataset.ytid;
  const iframe = document.createElement('iframe');
  iframe.src = `https://www.youtube.com/embed/${ytId}?rel=0&modestbranding=1&playsinline=1&autoplay=1`;
  iframe.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;border:none;';
  iframe.setAttribute('frameborder', '0');
  iframe.setAttribute('allow', 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture');
  iframe.setAttribute('allowfullscreen', '');
  thumb.replaceWith(iframe);
});


// ── Boot ──────────────────────────────────────────────────────────────────────
// Focus textarea on load
signInput.focus();
console.log('%c✋ handTalk Sign', 'font-size:16px;font-weight:bold;color:#29c49a');