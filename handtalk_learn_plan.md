# handTalk Learn — Full Implementation Plan
## Phase 3: Teaching Interface

---

## 1. Overview

handTalk Learn is a second service that sits alongside the existing Translator on the same
domain. It reuses the existing inference engine (FastAPI WebSocket + hoyso48 TFLite model)
without modification. No changes to main.py, app.js, or index.html are required.

The only new files are:
```
frontend/
├── learn.html          ← Learn page shell
├── learn.js            ← All Learn logic
└── learn.css           ← Learn-specific styles (imports shared tokens)
backend/
└── main.py             ← No changes. /vocab and /ws endpoints already exist.
```

The shared design system (green #29c49a, dark backgrounds, toast, camera overlay) is
inherited by importing the same CSS variables already defined in your main stylesheet.

---

## 2. Navigation Flow

### How users move between Translator and Learn

Both pages share a top navigation bar. The nav is a thin persistent strip that does NOT
auto-hide (unlike the translator's controls). It contains:

```
[ ✋ handTalk ]   [ Translate ↗ ]   [ Learn ↗ ]
```

On the Translator page (index.html), the existing topBar receives two nav links appended
to its right side. These links navigate to the other page — a full page load, not SPA
routing, which keeps both pages completely independent and prevents any MediaPipe/WASM
state from leaking across.

On the Learn page (learn.html), the same nav bar appears at the top. Below it, three
tabs select the active mode: Library | Practice | Quiz.

### Camera state handoff

When a user navigates from Translate → Learn, the browser disposes the Translate camera
stream naturally (page unload). Learn starts its own camera session fresh when the user
enters Practice or Quiz mode. Library mode never touches the camera.

This is intentional — it avoids the MediaPipe dual-model conflict documented in the
yoshan0921 reference app. Running Holistic and a separate gesture model simultaneously
caused WASM state corruption. By keeping pages separate and loading Holistic fresh on
each Learn session, this class of bug is eliminated.

---

## 3. Page Layout — learn.html

```
┌──────────────────────────────────────────────────────────┐
│  ✋ handTalk          [Translate]  [Learn]              │  ← nav bar (always visible)
├──────────────────────────────────────────────────────────┤
│                                                          │
│       [ Library ]   [ Practice ]   [ Quiz ]             │  ← mode tabs
│       ─────────────────────────────────────────         │
│                                                          │
│                  [MODE CONTENT AREA]                     │  ← swaps per tab
│                                                          │
└──────────────────────────────────────────────────────────┘
```

The mode content area is three separate `<div>` panels, only one visible at a time.
Switching tabs fades the old panel out and fades the new one in (150ms opacity transition).

---

## 4. Mode 1: Library

### Purpose
Browse all 250 signs with their YouTube demonstrations. No camera. Pure content.

### Layout
```
┌─ Library ──────────────────────────────────────────────┐
│  [🔍 Search signs...]          [All ▾] [category ▾]   │
│                                                         │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐        │
│  │  ▶   │ │  ▶   │ │  ▶   │ │  ▶   │ │  ▶   │        │
│  │hello │ │thank │ │mom   │ │dad   │ │good  │        │
│  │ ✓    │ │      │ │      │ │ ✓    │ │      │        │  ← ✓ = mastered
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘        │
│  ... (grid, 5 columns desktop / 2 mobile)              │
└────────────────────────────────────────────────────────┘
```

On card click → modal overlay opens:
```
┌─ Modal ────────────────────────────────────────────────┐
│  ✕                                    HELLO            │
│  ┌──────────────────────────────────────────────────┐  │
│  │          YouTube embed (autoplay muted)          │  │  ← iframe with yt_embedId
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  [ Practice this sign → ]                               │
└────────────────────────────────────────────────────────┘
```

The "Practice this sign →" button closes the modal, switches to Practice tab, and
pre-selects this sign as the target. This is the primary navigation path into Practice.

### Data source
```javascript
// On page load, fetch from existing /vocab endpoint
const res  = await fetch(`${API_BASE}/vocab`);
const data = await res.json();
// data.signs = [{sign_id, sign, yt_embedId}, ...]
```

### Filtering
Category mapping is a static JS object in learn.js — no backend change needed:

```javascript
const CATEGORIES = {
  family:   ["mom","dad","brother","sister","baby","family","grandma","grandpa","uncle","boy","girl","man","woman"],
  feelings: ["happy","sad","angry","scared","tired","love","hate","sorry","proud","funny","boring","hurt"],
  actions:  ["eat","drink","sleep","play","work","help","go","come","want","need","learn","read","write","swim","run","walk","drive"],
  objects:  ["book","phone","car","house","food","water","milk","ball","dog","cat","bird","fish","clock","chair","table"],
  greetings:["hello","thank-you","nice","meet","fine","good","morning","afternoon","night"],
  time:     ["today","tomorrow","yesterday","now","later","morning","night","week","year","time"],
  other:    [] // everything not in above buckets
};
```

### Mastered sign indicators
Signs in `localStorage.getItem('handTalk_progress')` that are marked mastered get a
green checkmark badge in the bottom-left of their card.

---

## 5. Mode 2: Practice

### Purpose
User selects a sign, watches the YouTube demo, then performs it on camera. The existing
WebSocket inference engine scores the attempt in real time. No new backend logic.

### Layout
```
┌─ Practice ─────────────────────────────────────────────┐
│  < Back to Library        Current: HELLO    [2/10 ✓]  │
│                                                         │
│  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │                     │  │                         │  │
│  │   YouTube Demo      │  │   Live Camera Feed      │  │
│  │   (iframe embed)    │  │   + landmark overlay    │  │
│  │                     │  │   + progress ring       │  │
│  └─────────────────────┘  └─────────────────────────┘  │
│                                                         │
│  Target: HELLO                                          │
│  ████████░░  Confidence: 72%     ● Listening...        │
│                                                         │
│  [ ✓ Got it — Next Sign ]    [ Skip ]                  │
└────────────────────────────────────────────────────────┘
```

On mobile, the two panels stack vertically (video top, camera bottom).

### Sign selection flow

Three entry points into Practice:
1. Library card click → "Practice this sign" → pre-selected
2. Practice tab → shows a sign picker grid (same cards as Library, no YouTube) → tap to select
3. Quiz "retry weak signs" button → queues a session of weak signs

### Live scoring mechanics

The existing WebSocket is connected when Practice starts. On every `prediction` message:

```javascript
function onPredictionMessage(msg) {
  // msg.sign    = what model thinks right now (or null if gated)
  // msg.top5    = [{sign, confidence}, ...]
  // msg.confidence = raw top prob

  const isTarget = msg.sign === currentTarget;

  // Update confidence bar — show even when not target (teaches user what NOT to do)
  updateConfBar(msg.confidence, isTarget);

  if (isTarget) {
    hitStreak++;
    if (hitStreak >= REQUIRED_HITS) {        // REQUIRED_HITS = 3 (~1.5s)
      markPassed(currentTarget);
      advanceToNextSign();
    }
  } else {
    hitStreak = 0;   // reset if wrong sign predicted
  }
}
```

`REQUIRED_HITS = 3` means the model must return the correct sign on 3 consecutive predict
cycles. Each cycle is ~0.5s (15 frames at 30fps), so the user holds the sign correctly
for about 1.5 seconds. This matches how the Kaggle model was designed — it evaluates
short isolated sign sequences, not single frames.

### Feedback states

| State | Visual |
|---|---|
| No hands detected | Camera ring pulses grey, "Show your hands" |
| Hands visible, wrong sign | Ring pulses orange, shows what model sees |
| Correct sign detected | Ring turns green, fills clockwise, confidence % shown |
| Sign confirmed (3 hits) | Ring completes → green flash → "✓ HELLO!" toast → advance |
| Gated out (low conf) | Ring is dim, shows gate reason from `msg.gate` |

### Session flow

A Practice session is a queue of signs. Default queue = 10 randomly selected signs from
the full 250 (weighted toward signs not yet mastered). The user can also build a custom
queue from the Library.

After each sign passes, the queue advances automatically. After all signs in the queue:

```
┌─ Session Complete ────────────────────────┐
│  🎉 Session done!                         │
│                                           │
│  ✓ 8 signs passed                        │
│  ✗ 2 signs struggled: GRANDMOTHER, UNCLE │
│                                           │
│  [Practice weak signs again]              │
│  [Go to Quiz]                             │
│  [Back to Library]                        │
└───────────────────────────────────────────┘
```

---

## 6. Mode 3: Quiz

### Purpose
Timed challenge. No video hint. User performs a sign from text prompt alone.
Tests retention, not just recognition.

### Layout
```
┌─ Quiz ─────────────────────────────────────────────────┐
│  Question 3 / 10                    ⏱ 8s               │
│  ─────────────────────────────────────────────────     │
│                                                         │
│                 Sign this word:                         │
│                                                         │
│                    HUNGRY                               │  ← large, centred
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Live Camera + landmark overlay         │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ████████░░░░░░░░░░  Timer bar (depletes left→right)   │
│                                                         │
│  Model sees: EAT (42%)                                  │  ← live top-1 shown
└────────────────────────────────────────────────────────┘
```

### Scoring rules

- Timer: 10 seconds per sign
- Pass: model confirms correct sign with 3 consecutive hits before timer expires
- Fail: timer expires OR user presses "Skip"
- Instant pass: if confidence ≥ 0.85 on first hit (allows fast signers to move quickly)

### Quiz results screen
```
┌─ Results ─────────────────────────────────────────────┐
│  Quiz Complete                         Score: 7 / 10  │
│                                                        │
│  ✓ hello        0.8s   ✓ dad          1.2s            │
│  ✓ happy        2.1s   ✗ grandmother  ✕ timeout       │
│  ✓ eat          0.9s   ✓ run          1.8s            │
│  ✗ uncle        ✕ skip ✓ want         2.4s            │
│  ✓ tired        1.5s   ✓ book         0.7s            │
│                                                        │
│  Streak: 🔥 5           Best time: 0.7s (book)        │
│                                                        │
│  [Try again]  [Practice weak signs]  [New quiz]       │
└───────────────────────────────────────────────────────┘
```

---

## 7. Data Structures

### localStorage schema
```javascript
// Key: 'handTalk_progress'
{
  version: 1,
  mastered: ["hello", "thank-you", "dad"],     // signed correctly 3+ times across sessions
  weak:     ["grandmother", "uncle"],           // failed in last quiz
  attempts: { "hello": 12, "hungry": 3 },      // total attempt count per sign
  bestTimes:{ "hello": 0.8, "book": 0.7 },     // fastest correct sign time in seconds
  streak:   7,                                  // current daily streak
  lastDate: "2026-04-01",                       // for streak calculation
  quizHistory: [
    { date: "2026-04-01", score: 7, total: 10, duration: 94 }
  ]
}
```

### In-memory session state (learn.js)
```javascript
const session = {
  mode:         'practice',   // 'library' | 'practice' | 'quiz'
  queue:        [],           // sign names in order
  queueIndex:   0,
  currentTarget: null,        // sign name being practiced now
  hitStreak:    0,
  startTime:    null,         // for timing individual signs
  results:      [],           // [{sign, passed, time, skipped}]
  ws:           null,         // WebSocket instance (null when Library active)
  holistic:     null,         // MediaPipe instance (null when Library active)
  cameraStream: null,
};
```

### Sign card data (from /vocab)
```javascript
// Fetched once on page load, cached in memory
let VOCAB = [];  // [{sign_id, sign, yt_embedId}]

// Augmented at runtime with progress data
function enrichedSign(entry) {
  const prog = progress.attempts[entry.sign] || 0;
  return {
    ...entry,
    mastered: progress.mastered.includes(entry.sign),
    weak:     progress.weak.includes(entry.sign),
    attempts: prog,
    category: getCategory(entry.sign),   // from CATEGORIES map
  };
}
```

---

## 8. WebSocket Reuse Strategy

### Connection lifecycle in Learn

```
Library tab active:
  → ws = null (no connection, camera off)
  → Saves battery, no unnecessary inference

Practice/Quiz tab activated:
  → initHolistic() called (same CDN scripts as Translate)
  → getUserMedia() called
  → connectLearnWS() opens ws to /ws/{learnClientId}
  → learnClientId = 'learn_' + Math.random().toString(36).slice(2)
     (different from Translate CLIENT_ID — prevents session collision)

Practice/Quiz tab deactivated OR user navigates back to Library:
  → ws.close()
  → cameraStream.getTracks().forEach(t => t.stop())
  → holistic instance released
  → All cleaned up before switching
```

### Message handling in Learn vs Translate

The WebSocket protocol is identical. Learn uses the same `frame` / `predict` / `flush`
actions. The only difference is what the `onmessage` handler does with the predictions:

```javascript
// Translate onmessage: display sign in HUD, build sentences
// Learn onmessage: score against currentTarget

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'prediction') {
    if (session.mode === 'practice' || session.mode === 'quiz') {
      scorePrediction(msg);   // learn-specific handler
    }
  }
};
```

No backend changes. No new WebSocket endpoints. The same inference stream is repurposed
purely by changing what the frontend does with the messages.

### set_threshold for Learn mode

When entering Practice/Quiz, send tighter gate thresholds to the backend:

```javascript
// Translate defaults: confidence=0.22, margin=0.04
// Learn mode: tighten to reduce false positives during scoring
ws.send(JSON.stringify({
  action: 'set_threshold',
  confidence: 0.45,    // higher — we only want clear predictions
  margin: 0.10,        // higher — must be unambiguously one sign
  consecutive: 3,      // match REQUIRED_HITS
}));
```

When leaving Learn mode (tab close or nav away), `ws.close()` disposes the session.
The next Translate session opens a fresh connection with default thresholds.

---

## 9. Visual Style — Consistency with Translator

### CSS variables inherited unchanged
```css
/* These already exist in your main stylesheet — Learn imports them */
--color-accent:   #29c49a;
--color-accent-2: #4dd9b4;
--color-bg:       #0a0a0f;
--color-surface:  #14141f;
--color-border:   rgba(41,196,154,.15);
--color-text:     #e8e8ec;
--color-dim:      #7a7a8e;
--radius-card:    12px;
--font-main:      /* whatever your translator uses */
```

### Learn-specific additions (in learn.css)
```css
/* Progress ring on camera feed */
.learn-ring {
  border: 3px solid var(--color-border);
  border-radius: 50%;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.learn-ring.active   { border-color: var(--color-accent); }
.learn-ring.success  { border-color: var(--color-accent); box-shadow: 0 0 20px rgba(41,196,154,.4); }

/* Sign card grid */
.sign-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-card);
  cursor: pointer;
  transition: border-color 0.15s, transform 0.15s;
}
.sign-card:hover          { border-color: var(--color-accent); transform: translateY(-2px); }
.sign-card.mastered::after { content: '✓'; color: var(--color-accent); }

/* Confidence bar */
.conf-bar-fill { background: var(--color-accent); transition: width 0.1s linear; }

/* Timer bar (quiz) */
.timer-bar-fill { background: var(--color-accent); transition: width 1s linear; }
.timer-bar-fill.urgent { background: #e05a5a; }   /* red when < 3s */

/* Mode tabs */
.learn-tab.active { border-bottom: 2px solid var(--color-accent); color: var(--color-accent); }
```

### Toast reuse
The same `toast()` function from app.js is copy-included in learn.js (or extracted to a
shared `utils.js`). Same animation, same positioning, same styling. Learn uses it for:
- "✓ HELLO — passed!" (green, 1.5s)
- "✕ Time's up" (red, 2s)
- "Session saved" (neutral, 2s)

---

## 10. File Structure

```
frontend/
├── index.html          ← Translator (existing, minor nav addition only)
├── app.js              ← Translator logic (no changes)
├── style.css           ← Shared CSS variables (no changes, maybe minor nav addition)
├── learn.html          ← NEW: Learn page shell
├── learn.js            ← NEW: Library + Practice + Quiz logic (~600 lines)
└── learn.css           ← NEW: Learn-specific styles (~200 lines)

backend/
└── main.py             ← NO CHANGES
```

### What changes in index.html (the only touch to existing files)
Add two lines to the existing topBar:
```html
<nav class="ht-nav">
  <a href="index.html" class="ht-nav-link active">Translate</a>
  <a href="learn.html" class="ht-nav-link">Learn</a>
</nav>
```
That's the only modification to any existing file.

---

## 11. Build Order

Build in this sequence to keep each step independently testable:

1. **learn.html + learn.css skeleton** — nav, tabs, empty panels, style tokens wired up
2. **Library mode** — fetch /vocab, render grid, search/filter, modal with YouTube embed
3. **Practice mode infrastructure** — camera init, WebSocket connect, landmark overlay
4. **Scoring engine** — `scorePrediction()`, hit streak, pass/fail logic, progress ring
5. **Practice session flow** — queue, advance, session complete screen
6. **localStorage progress** — mastered/weak/streaks persist and reflect in Library cards
7. **Quiz mode** — timer, text prompt, same scoring engine, results screen
8. **Polish** — transitions, feedback states, mobile layout, toast messages

Each step produces a working, testable increment. Library is useful standalone (no camera).
Practice is useful before Quiz is built. The scoring engine is shared between both.

---

## 12. What Is NOT Changing

To be explicit about what stays untouched:

- `main.py` — no new endpoints, no new WebSocket actions, no logic changes
- `index.html` — only the two nav links added, nothing else
- `app.js` — not modified at all
- `style.css` — not modified (learn.css imports tokens from it)
- The inference pipeline — identical in Learn and Translate
- The MediaPipe Holistic setup — same CDN scripts, same options
- The `frame` / `predict` / `flush` WebSocket protocol — unchanged

The Learn module is purely additive. It can be removed by deleting three files and two
nav links, with zero impact on the Translator.
