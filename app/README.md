# Personality Prediction — Web App

A small web app that predicts Big Five (OCEAN) personality traits from text,
served by this repo's pretrained Bag-of-Words + scikit-learn models
(`data/models/`). Paste a paragraph, get per-trait scores and a radar chart.

The web server uses only the Python standard library. scikit-learn is needed
only to load the pretrained models.

## Run

```bash
# 1. install the ML stack that can unpickle the models
pip install -r app/requirements.txt

# 2. start the app
python app/server.py            # http://127.0.0.1:8000
# options: --host 0.0.0.0 --port 5000
```

Open the printed URL, paste text (or click **Load sample**), and hit **Analyze**.

## API

`POST /predict` with JSON `{"text": "..."}` returns:

```json
{
  "sentences": 3,
  "traits": [
    {"code": "EXT", "name": "Extraversion", "score": 1.0, "percent": 100.0, "label": "high"},
    ...
  ]
}
```

`GET /health` returns `{"ok": true}`.

## Notes

- Each trait has its own binary classifier (Logistic Regression, except NEU
  which is a Random Forest) over Bag-of-Words features. Text is split into
  sentences; `score` is the fraction of sentences flagged positive.
- Longer passages give more stable scores. Treat results as rough tendencies,
  not clinical facts (accuracy ~61–80% per trait; see the top-level `README.md`).
- The models were pickled with scikit-learn 0.22.1, so `app/requirements.txt`
  pins scikit-learn 1.2.2 / numpy<2 to unpickle them (the NEU RandomForest
  needs a pre-1.3 tree format).
