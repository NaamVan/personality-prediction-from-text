---
name: personality-prediction
description: Predict Big Five (OCEAN) personality traits from a passage of text using this repo's pretrained Bag-of-Words + scikit-learn models. Use when the user wants to analyze writing, an essay, a message, or any text sample for Extraversion, Neuroticism, Agreeableness, Conscientiousness, and Openness — e.g. "what personality does this text show", "run a Big Five / OCEAN analysis on this", "predict the author's traits".
---

# Personality Prediction from Text

Predict the author's **Big Five** (OCEAN) personality traits from a text
sample, using the pretrained models shipped in this repository
(`data/models/`). No training or downloads are needed for inference.

The five traits:

| Code | Trait             |
|------|-------------------|
| EXT  | Extraversion      |
| NEU  | Neuroticism       |
| AGR  | Agreeableness     |
| CON  | Conscientiousness |
| OPN  | Openness          |

Each trait has its own binary classifier (Logistic Regression, except NEU
which is a Random Forest) applied to Bag-of-Words features. The text is split
into sentences, every sentence is classified, and the skill reports the share
of sentences predicted positive (a soft score in `[0, 1]`) plus a majority
verdict per trait.

## How to run it

The models were pickled with an old scikit-learn (0.22.1). To unpickle them —
the NEU Random Forest in particular — install the pinned versions once:

```bash
pip install -r .claude/skills/personality-prediction/scripts/requirements.txt
```

Then predict. Text can be passed as an argument, from a file, or on stdin:

```bash
# argument
python .claude/skills/personality-prediction/scripts/predict.py "Your text here."

# from a file
python .claude/skills/personality-prediction/scripts/predict.py --file sample.txt

# from stdin
cat sample.txt | python .claude/skills/personality-prediction/scripts/predict.py

# machine-readable output
python .claude/skills/personality-prediction/scripts/predict.py "Your text." --json
```

Example output:

```
Big Five personality prediction
==================================
EXT  Extraversion        66.7%  #############        high
NEU  Neuroticism        100.0%  #################### high
AGR  Agreeableness       33.3%  #######              low
CON  Conscientiousness   33.3%  #######              low
OPN  Openness           100.0%  #################### high
```

## Guidance for interpreting results

- Longer passages (several sentences) give more stable scores; a single short
  sentence yields a coarse 0/1 per trait.
- Report scores as tendencies, not clinical facts. These are noisy models
  trained on essay/forum/Reddit text (see `README.md` for accuracy: roughly
  61–80% per trait, NEU being weakest).
- The score is the fraction of sentences a trait's classifier flagged
  positive — treat it as a rough confidence, not a calibrated probability.

## Under the hood

Mirrors `predict.ipynb`: `cNEU` uses `vectorizer_30.p`; the other four
classifiers use `vectorizer_31.p`. See `README.md` for data sources, method,
and per-trait accuracy. To retrain or use the GloVe variant instead of
Bag-of-Words, follow the notebook workflow in the README
(`preprocessing.ipynb` → `model_glove.ipynb` / `model_bow.ipynb`).
