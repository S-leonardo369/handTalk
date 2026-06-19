---
name: ASL Accuracy and Teaching Roadmap
overview: Implement Milestone 1 as recognition accuracy + reliability, then add bidirectional voice/sign features and a teaching interface that reuses your existing sign map YouTube assets.
todos:
  - id: accuracy-instrumentation
    content: Add inference diagnostics and gate counters in backend API/WebSocket flow
    status: in_progress
  - id: temporal-stability
    content: Implement prediction smoothing and configurable commit/suggestion thresholds
    status: pending
  - id: retraining-pipeline
    content: Set up light retraining/evaluation scripts and export updated TFLite artifacts
    status: pending
  - id: voice-bidirectional
    content: Implement sign->voice controls and voice->sign transcript mapping via vocab_map
    status: pending
  - id: teaching-module
    content: Create teaching/practice/quiz frontend module using YouTube sign assets
    status: pending
  - id: autocomplete
    content: Add top-k suggestion pipeline and frontend acceptance interactions
    status: pending
isProject: false
---

# ASL Accuracy + Voice + Teaching Plan

## Goals and Scope

- Milestone 1: maximize recognition accuracy first, including light retraining.
- Then add both directions of voice/sign:
  - sign -> voice (TTS for recognized output)
  - voice -> sign (speech input mapped to known sign vocabulary/videos)
- Add a teaching interface inspired by the referenced app, using existing `vocab_map.json` YouTube links.
- Add sign auto-complete suggestions for partial/uncertain predictions.

## Phase 1: Accuracy First (Backend + Frontend + Data)

- Validate and lock inference contract in [backend/main.py](C:/Users/Bazuka/Documents/handTalk/backend/main.py):
  - strict `serving_default` signature
  - strict `inputs`/`outputs` key checks
  - shape guard `(n_frames, ROWS_PER_FRAME, 3)`
  - dynamic `ROWS_PER_FRAME` from `vocab_format.parquet`
- Add runtime diagnostics endpoint(s) in [backend/main.py](C:/Users/Bazuka/Documents/handTalk/backend/main.py):
  - active gates
  - model signature info
  - frame acceptance/drop reasons counters (no hands, low motion, confidence gate, margin gate)
- Improve temporal stability in [backend/main.py](C:/Users/Bazuka/Documents/handTalk/backend/main.py):
  - median/EMA smoothing over recent predictions
  - optional dynamic thresholds per sign frequency/confusion
- Frontend reliability and calibration UI in [frontend/app.js](C:/Users/Bazuka/Documents/handTalk/frontend/app.js):
  - expose live confidence/margin and gate rejection reason
  - per-session threshold tuning sliders persisted in local storage
- Light retraining workflow:
  - create `training/` scripts (or adapt existing) for fine-tuning with your captured data
  - export updated `backend/model/vocab_model_hoyso48.tflite`, `vocab_map.json`, `vocab_format.parquet`
  - add evaluation report output (top-1 accuracy, confusion matrix, per-sign recall)

## Phase 2: Voice <-> Sign

- Sign -> voice in [frontend/app.js](C:/Users/Bazuka/Documents/handTalk/frontend/app.js):
  - keep/improve Web Speech TTS trigger policy (speak sentence on flush, optional speak-top-sign mode)
  - add controls for voice/rate/pitch and mute toggle
- Voice -> sign:
  - use browser SpeechRecognition in [frontend/app.js](C:/Users/Bazuka/Documents/handTalk/frontend/app.js)
  - normalize transcript text and map words/phrases to known signs from [backend/model/vocab_map.json](C:/Users/Bazuka/Documents/handTalk/backend/model/vocab_map.json)
  - show matched sign cards with YouTube embed links
- Add backend helper endpoint in [backend/main.py](C:/Users/Bazuka/Documents/handTalk/backend/main.py):
  - `/sign-search?q=` for canonical matching and alias/fuzzy matching support

## Phase 3: Teaching Interface

- Add dedicated teaching page/module under frontend (e.g., `frontend/teaching.html` + `frontend/teaching.js`):
  - sign library browser
  - practice mode (show target sign video, user performs sign, live scoring)
  - quiz mode (multiple choice + perform-and-check)
- Reuse websocket inference stream for scoring:
  - compare expected sign vs top-k predictions + confidence over short window
  - provide actionable feedback ("hold longer", "move clearer", "close hand shape")
- Track simple progress locally:
  - mastered signs, streaks, weak signs list

## Phase 4: Auto-complete Signs

- Add auto-complete logic for uncertain frames in [backend/main.py](C:/Users/Bazuka/Documents/handTalk/backend/main.py):
  - when confidence below commit threshold but above suggestion threshold, send top-k suggestions
  - include language-model-like ranking from recent signed context (n-gram over sign history)
- UI suggestions in [frontend/app.js](C:/Users/Bazuka/Documents/handTalk/frontend/app.js):
  - chip list of likely next signs
  - one-click accept suggestion to speed sentence completion

## Validation and Success Metrics

- Recognition quality:
  - top-1 and top-3 accuracy on validation clips
  - per-sign recall for most-used signs
  - false-positive rate during idle/no-sign periods
- UX performance:
  - median prediction latency
  - websocket reconnect success rate
  - end-to-end sentence completion time

## Delivery Order (recommended)

1. Accuracy instrumentation + smoothing
2. Light retraining + model refresh
3. Voice <-> sign features
4. Teaching interface
5. Auto-complete suggestions

