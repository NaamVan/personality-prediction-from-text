#!/usr/bin/env python
"""Predict Big Five personality traits from text using the repo's pretrained
Bag-of-Words + Logistic Regression models.

Usage:
    python predict.py "your text here"
    python predict.py --file path/to/text.txt
    echo "your text" | python predict.py
    python predict.py "text" --json

The five traits (OCEAN / Big Five):
    EXT  Extraversion
    NEU  Neuroticism
    AGR  Agreeableness
    CON  Conscientiousness
    OPN  Openness

Each pretrained classifier predicts a binary label per sentence. This script
reports, for each trait, the share of sentences predicted positive (a soft
score in [0, 1]) and a majority-vote binary verdict.
"""
import argparse
import json
import os
import pickle
import re
import sys
import warnings

# Pickles were saved with scikit-learn 0.22.1; silence the version-mismatch
# noise so the output stays clean. Predictions remain valid for these simple
# linear models.
warnings.filterwarnings("ignore")

TRAITS = ["EXT", "NEU", "AGR", "CON", "OPN"]
TRAIT_NAMES = {
    "EXT": "Extraversion",
    "NEU": "Neuroticism",
    "AGR": "Agreeableness",
    "CON": "Conscientiousness",
    "OPN": "Openness",
}

# Resolve the repo's data/models directory relative to this file:
# .claude/skills/personality-prediction/scripts/predict.py -> repo root is 4 up.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "data", "models")


def _load(name):
    with open(os.path.join(MODELS_DIR, name), "rb") as fh:
        return pickle.load(fh)


def load_models():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit(
            "scikit-learn is required. Install it with:\n"
            "    pip install scikit-learn numpy scipy"
        )
    if not os.path.isdir(MODELS_DIR):
        sys.exit(f"Models directory not found: {MODELS_DIR}")

    # cNEU uses vectorizer_30; the other four use vectorizer_31 (matching the
    # feature space each classifier was trained on, per predict.ipynb).
    clfs = {t: _load(f"c{t}.p") for t in TRAITS}
    v31 = _load("vectorizer_31.p")
    v30 = _load("vectorizer_30.p")
    vecs = {"EXT": v31, "NEU": v30, "AGR": v31, "CON": v31, "OPN": v31}
    return clfs, vecs


def split_sentences(text):
    sentences = [s for s in re.split(r"(?<=[.!?]) +", text.strip()) if s.strip()]
    return sentences or [text.strip()]


def predict_personality(text, clfs, vecs):
    sentences = split_sentences(text)
    result = {}
    for t in TRAITS:
        X = vecs[t].transform(sentences)
        preds = clfs[t].predict(X)
        score = float(sum(int(p) for p in preds)) / len(preds)
        result[t] = {
            "name": TRAIT_NAMES[t],
            "score": round(score, 4),
            "label": bool(round(score)),
        }
    result["_sentences"] = len(sentences)
    return result


def render_text(result):
    n = result.pop("_sentences", None)
    lines = ["Big Five personality prediction", "=" * 34]
    for t in TRAITS:
        r = result[t]
        bar = "#" * int(round(r["score"] * 20))
        verdict = "high" if r["label"] else "low"
        lines.append(
            f"{t}  {r['name']:<18} {r['score'] * 100:5.1f}%  "
            f"{bar:<20} {verdict}"
        )
    if n is not None:
        lines.append("")
        lines.append(f"(scored across {n} sentence(s))")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="Text to analyze")
    ap.add_argument("--file", help="Read text from a file instead")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    args = ap.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        ap.error("provide text as an argument, via --file, or on stdin")

    if not text.strip():
        ap.error("no text provided")

    clfs, vecs = load_models()
    result = predict_personality(text, clfs, vecs)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_text(result))


if __name__ == "__main__":
    main()
