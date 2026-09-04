# HEXACO-60 Personality Self-Test (web app)

A single-file, self-contained web app that administers and scores the
**HEXACO-PI-R (60-item)** self-report personality inventory. No server, no
build step, no dependencies to install — just open the file.

## Use it

Open `hexaco/index.html` in any modern browser (double-click it, or serve the
folder). Answer the 60 statements on the 1–5 scale, then click **See my
results**.

You get:

- **Six factor scores** (mean 1–5 after reverse-keying), each with a bar, the
  norm-group average marked on it, a band (Low / Below average / Average /
  Above average / High) and an approximate percentile.
- **24 facet scores** under each factor (expand "Facets").
- A **radar chart** across the six factors (uses Chart.js from a CDN; the bar
  scores work fully offline even if the chart can't load).
- A norm-group selector — compare against the college-sample norms for
  everyone, women, or men.
- **Print / save as PDF** of the results.

## The six factors

`H` Honesty-Humility · `E` Emotionality · `X` Extraversion ·
`A` Agreeableness · `C` Conscientiousness · `O` Openness to Experience

## How scoring works

- Each factor is the mean of its 10 items; each of the 24 facets is the mean of
  its 2–3 items.
- Reverse-keyed items (per the official scoring keys) are recoded
  `5→1, 4→2, 3→3, 2→4, 1→5` before averaging.
- Norm comparison uses the self-report means and standard deviations from the
  HEXACO-60 college-student sample (N=1126), converting your score to a
  z-score → band and percentile.

The item text, scoring keys, and norms come from the official HEXACO-60
materials (`English_self60`, `ScoringKeys_60`, `descriptives_60`).

## Notes / credit

- Instrument: **HEXACO-PI-R © Kibeom Lee & Michael C. Ashton** — see
  <https://hexaco.org>. Provided here for research/educational use.
- This is an **unofficial scorer for self-exploration, not a clinical or
  diagnostic assessment**. The 60-item facet scales are short by design and
  are meant as rough indicators only.
- HEXACO is a different framework from the Big Five text-prediction model in
  the rest of this repository: it has six factors (it adds Honesty-Humility)
  and is filled in by rating statements rather than by analyzing free text.
