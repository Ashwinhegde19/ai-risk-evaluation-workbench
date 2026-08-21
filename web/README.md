# VERDICT — evidence console

Static front-end for the AI-safety evaluation platform: a forensic red-team
dossier over the seed-42 canonical run (15 strategies × 5 seeded trials,
LLM-adjudicated breaks with 95% Wilson confidence intervals).

## Run it

```bash
cd web && python3 -m http.server 8000
# → http://localhost:8000
```

No build step, no dependencies — vanilla HTML/CSS/JS. Opening
`index.html` directly (`file://`) also works: the page falls back to the
inlined seed-42 dataset (the exact numbers in `results/`) whenever
`fetch()` is unavailable, and upgrades to the live JSON when served.

## Data

The page loads, via `fetch()`:

- `results/redteam_findings.json` — per-model / per-strategy break rates,
  Wilson CIs, and all 300 case files with full transcripts
- `results/compliance_report_model.json` — risk tiers, certificates, and
  the 79 framework citations (EU AI Act / NIST AI RMF / ISO 42001)

`web/results` is a symlink to `../results` so the data stays inside the
server root when serving from `web/`. Every number rendered traces to a
field in those two files; the inlined `DATA` constant in `assets/app.js`
is only the offline fallback.

## Note

`src/dashboard/app.py` (the Streamlit quick-look) is intentionally left
intact — this console is a separate, static artifact and does not replace it.
