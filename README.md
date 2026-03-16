# ASL Real-Time Sign Language Translator

A full-stack web application that translates American Sign Language into English sentences using your webcam, MediaPipe, and a trained LSTM model.

---

## Project Structure

```
asl-translator/
  frontend/               ← Browser app (HTML + CSS + JS)
    index.html
    style.css
    app.js
  backend/                ← FastAPI server
    main.py
    Dockerfile
    model/                ← Trained model goes here (after training)
  data/
    raw/                  ← Collected JSON samples go here
  training/
    collect_data.py       ← Step 1: Collect sign data
    train_model.py        ← Step 2: Train the model
  requirements.txt
  README.md
```

---

## Quick Start (3 steps)

### Step 1 — Install dependencies

```bash
# Create and activate a virtual environment
python -m venv asl_env
source asl_env/bin/activate      # Mac/Linux
# asl_env\Scripts\activate       # Windows

# Install packages
pip install -r requirements.txt
```

### Step 2 — Collect training data

```bash
cd training
python collect_data.py
```

**Controls:**
- `SPACE` — start / stop recording a sample
- `S`     — save the current sample
- `D`     — discard the current sample
- `N`     — move to the next sign
- `Q`     — quit

Collect at least **100 samples per sign** for acceptable accuracy. More is better.
Vary lighting, distance from camera, and if possible collect from multiple people.

### Step 3 — Train the model

```bash
cd training
python train_model.py
```

This saves:
- `backend/model/asl_model.h5`    — the trained model
- `backend/model/label_map.json`  — maps class indices to sign names
- `backend/model/training_plot.png` — accuracy/loss curves

Training takes 5–20 minutes depending on your hardware and dataset size.

### Step 4 — Start the backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see: `Model loaded — N signs`

### Step 5 — Open the frontend

Serve the frontend with any static file server. The easiest:

```bash
cd frontend
python -m http.server 3000
```

Then open: **http://localhost:3000**

> ⚠️ Do NOT open index.html directly as a file:// URL.
> Camera access (getUserMedia) requires HTTP or HTTPS, not file://.

---

## How It Works

```
Browser                              FastAPI Backend
─────────────────────────────        ─────────────────────────────
Camera (getUserMedia)
  ↓
MediaPipe Tasks (GPU)           
  → 21 hand landmarks per frame
  ↓
Normalise (centre on wrist,
  scale by wrist→tip distance)
  ↓
Sliding window: last 30 frames  →→→  /ws/{id} WebSocket
                                        ↓
                                      LSTM model predicts sign
                                        ↓
                                      Sign buffer + pause detection
                                        ↓
                                      Rule-based gloss→English
                                      (swap for LLM when ready)
  ↓  ←←←←←←←←←←←←←←←←←←←←←←←←←←←
Web Speech API (TTS)
Display sentence in UI
```

---

## Tuning Tips

### Confidence threshold (UI slider)
- **72%** — default, balanced
- **82%** — fewer false positives, may miss quick signs
- **65%** — more sensitive, more false positives

### Pause-to-flush threshold (UI slider)
- Controls how long a gap between signs triggers a sentence flush
- Default: **1.5 seconds**
- Raise it if you tend to pause between signs naturally

### Improving accuracy
1. Collect more samples — especially for signs you see confused
2. Collect from multiple people with different hand sizes
3. Collect under different lighting conditions
4. After retraining, restart the backend (it reloads the model on startup)

---

## Adding the LLM Sentence Translator (Optional)

When you have an Anthropic API key, replace the `/translate` endpoint in `backend/main.py`:

```python
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

@app.post("/translate")
def translate(req: TranslateRequest):
    gloss = " ".join(req.signs)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": (
                f"Convert this ASL gloss to a natural English sentence. "
                f"Output only the sentence, nothing else. Gloss: {gloss}"
            )
        }]
    )
    return {"sentence": msg.content[0].text, "gloss": gloss}
```

Set your key: `export ANTHROPIC_API_KEY=sk-ant-...`

---

## Deployment

### Backend → Render.com
1. Push `backend/` folder to a GitHub repo
2. Create a new Web Service on render.com
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add env var `ANTHROPIC_API_KEY` if using LLM translation

### Frontend → Vercel / Netlify
1. Update `API_BASE` and `WS_BASE` in `frontend/app.js` to your Render URL
2. Push `frontend/` to GitHub and connect to Vercel
3. Deploy — Vercel hosts it for free with HTTPS

---

## Signs Included (default vocabulary)

Greetings: HELLO, GOODBYE, PLEASE, THANK-YOU, SORRY  
Pronouns:  I, YOU, HE, SHE, WE, THEY  
Verbs:     WANT, NEED, HAVE, GO, COME, HELP, LIKE, KNOW, SEE  
Questions: WHAT, WHERE, WHEN, WHO, WHY  
Responses: YES, NO, UNDERSTAND  
Descriptors: GOOD, BAD, MORE  

Add more by editing `SIGNS_TO_COLLECT` in `training/collect_data.py`.

---

## Requirements

- Python 3.10 or 3.11
- Webcam
- Chrome or Edge (recommended for MediaPipe GPU delegate)
- ~4GB RAM for training
