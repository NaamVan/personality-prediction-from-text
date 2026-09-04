#!/usr/bin/env python
"""Personality Prediction web app.

A dependency-light web app (Python standard library only for the server) that
serves a browser UI and a JSON API for predicting Big Five (OCEAN) personality
traits from text, using this repo's pretrained Bag-of-Words + scikit-learn
models in ``data/models/``.

Run:
    python app/server.py            # then open http://127.0.0.1:8000
    python app/server.py --port 5000 --host 0.0.0.0

Requires the ML stack that can unpickle the models (see app/requirements.txt):
    pip install -r app/requirements.txt
"""
import argparse
import json
import os
import pickle
import re
import sys
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

warnings.filterwarnings("ignore")

TRAITS = ["EXT", "NEU", "AGR", "CON", "OPN"]
TRAIT_NAMES = {
    "EXT": "Extraversion",
    "NEU": "Neuroticism",
    "AGR": "Agreeableness",
    "CON": "Conscientiousness",
    "OPN": "Openness",
}

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "data", "models")

_STATE = {"clfs": None, "vecs": None}


def _load(name):
    with open(os.path.join(MODELS_DIR, name), "rb") as fh:
        return pickle.load(fh)


def load_models():
    """Load the 5 classifiers and 2 vectorizers once, lazily."""
    if _STATE["clfs"] is not None:
        return _STATE["clfs"], _STATE["vecs"]
    try:
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit(
            "scikit-learn is required. Install it with:\n"
            "    pip install -r app/requirements.txt"
        )
    if not os.path.isdir(MODELS_DIR):
        sys.exit(f"Models directory not found: {MODELS_DIR}")

    clfs = {t: _load(f"c{t}.p") for t in TRAITS}
    v31 = _load("vectorizer_31.p")
    v30 = _load("vectorizer_30.p")
    # cNEU was trained against vectorizer_30; the rest against vectorizer_31.
    vecs = {"EXT": v31, "NEU": v30, "AGR": v31, "CON": v31, "OPN": v31}
    _STATE["clfs"], _STATE["vecs"] = clfs, vecs
    return clfs, vecs


def split_sentences(text):
    sentences = [s for s in re.split(r"(?<=[.!?]) +", text.strip()) if s.strip()]
    return sentences or [text.strip()]


def predict_personality(text):
    clfs, vecs = load_models()
    sentences = split_sentences(text)
    traits = []
    for t in TRAITS:
        X = vecs[t].transform(sentences)
        preds = clfs[t].predict(X)
        score = float(sum(int(p) for p in preds)) / len(preds)
        traits.append({
            "code": t,
            "name": TRAIT_NAMES[t],
            "score": round(score, 4),
            "percent": round(score * 100, 1),
            "label": "high" if round(score) else "low",
        })
    return {"sentences": len(sentences), "traits": traits}


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Personality Prediction from Text</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root { color-scheme: light dark; --bg:#f6f7f9; --card:#fff; --fg:#1a1d24;
    --muted:#5b6472; --accent:#4f6ef7; --border:#e3e6eb; }
  @media (prefers-color-scheme: dark){ :root{ --bg:#12151b; --card:#1b1f27;
    --fg:#e8eaed; --muted:#9aa3b2; --accent:#7c92ff; --border:#2a2f3a; } }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg); color:var(--fg); }
  .wrap { max-width:860px; margin:0 auto; padding:32px 20px 64px; }
  h1 { font-size:24px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 24px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px;
    padding:20px; margin-bottom:20px; }
  textarea { width:100%; min-height:150px; resize:vertical; padding:12px;
    border:1px solid var(--border); border-radius:8px; background:transparent;
    color:var(--fg); font:inherit; }
  .row { display:flex; gap:10px; align-items:center; margin-top:12px; flex-wrap:wrap; }
  button { background:var(--accent); color:#fff; border:0; border-radius:8px;
    padding:10px 18px; font:inherit; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.6; cursor:default; }
  .ghost { background:transparent; color:var(--accent); border:1px solid var(--border); }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  @media (max-width:680px){ .grid{ grid-template-columns:1fr; } }
  .bars { display:flex; flex-direction:column; gap:12px; }
  .bar-row { display:grid; grid-template-columns:130px 1fr 48px; align-items:center; gap:10px; }
  .bar-track { background:var(--border); border-radius:99px; height:10px; overflow:hidden; }
  .bar-fill { background:var(--accent); height:100%; border-radius:99px; transition:width .5s ease; }
  .pct { text-align:right; color:var(--muted); font-variant-numeric:tabular-nums; }
  .tname { color:var(--fg); } .tname small { color:var(--muted); }
  .hint { color:var(--muted); font-size:13px; margin-top:14px; }
  .err { color:#d9534f; margin-top:12px; }
  #meta { color:var(--muted); font-size:13px; margin-top:8px; }
  a { color:var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Personality Prediction from Text</h1>
  <p class="sub">Predicts the author's Big Five (OCEAN) traits using pretrained models.</p>

  <div class="card">
    <textarea id="text" placeholder="Paste a paragraph of writing here — an essay, a message, a post…"></textarea>
    <div class="row">
      <button id="go">Analyze</button>
      <button id="sample" class="ghost">Load sample</button>
    </div>
    <div id="err" class="err" hidden></div>
  </div>

  <div id="results" class="card" hidden>
    <div class="grid">
      <div>
        <div class="bars" id="bars"></div>
        <div id="meta"></div>
      </div>
      <div><canvas id="radar" height="280"></canvas></div>
    </div>
    <p class="hint">Scores are the share of sentences each trait's classifier flagged
      positive — a rough tendency, not a clinical measure. Accuracy is roughly 61–80%
      per trait (Neuroticism weakest).</p>
  </div>
</div>

<script>
const SAMPLE = "I love going to parties and meeting new people. It is always exciting to be around a crowd. Still, I plan my week carefully and keep things organized. I am fascinated by new ideas and abstract concepts, and I try to be kind and helpful to everyone I meet.";
const $ = id => document.getElementById(id);
let chart = null;

$('sample').onclick = () => { $('text').value = SAMPLE; };

async function analyze(){
  const text = $('text').value.trim();
  $('err').hidden = true;
  if(!text){ $('err').textContent = "Please enter some text."; $('err').hidden = false; return; }
  $('go').disabled = true; $('go').textContent = "Analyzing…";
  try{
    const res = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})});
    if(!res.ok) throw new Error('Server error ' + res.status);
    const data = await res.json();
    render(data);
  }catch(e){
    $('err').textContent = e.message; $('err').hidden = false;
  }finally{
    $('go').disabled = false; $('go').textContent = "Analyze";
  }
}
$('go').onclick = analyze;

function render(data){
  $('results').hidden = false;
  const bars = $('bars'); bars.innerHTML = '';
  for(const t of data.traits){
    const row = document.createElement('div'); row.className = 'bar-row';
    row.innerHTML = `<div class="tname">${t.name} <small>(${t.code})</small></div>
      <div class="bar-track"><div class="bar-fill" style="width:${t.percent}%"></div></div>
      <div class="pct">${t.percent}%</div>`;
    bars.appendChild(row);
  }
  $('meta').textContent = `Scored across ${data.sentences} sentence(s).`;

  const labels = data.traits.map(t => t.code);
  const values = data.traits.map(t => t.percent);
  const style = getComputedStyle(document.documentElement);
  const accent = style.getPropertyValue('--accent').trim();
  const grid = style.getPropertyValue('--border').trim();
  const fg = style.getPropertyValue('--muted').trim();
  if(chart) chart.destroy();
  chart = new Chart($('radar'), {
    type:'radar',
    data:{ labels, datasets:[{ label:'Score %', data:values,
      borderColor:accent, backgroundColor:accent+'33', pointBackgroundColor:accent }]},
    options:{ responsive:true, scales:{ r:{ min:0, max:100,
      grid:{color:grid}, angleLines:{color:grid}, pointLabels:{color:fg},
      ticks:{display:false} } }, plugins:{ legend:{display:false} } }
  });
}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = (payload.get("text") or "").strip()
            if not text:
                self._send(400, json.dumps({"error": "no text provided"}))
                return
            self._send(200, json.dumps(predict_personality(text)))
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(e)}))

    def log_message(self, fmt, *args):  # keep the console quiet
        return


def main():
    ap = argparse.ArgumentParser(description="Personality Prediction web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print("Loading models…")
    load_models()
    print(f"Ready. Open http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
