# Changelog

Dated record of substantive work on this project. Each entry is what changed and
why — not a commit-by-commit log (see `git log` for that).

## 2026-08-03 — audit + honest-results refresh

- Pushed a 7-week-old local-only commit (README "Why this exists" positioning +
  Cloud Run framing) that had never reached `origin/main` — the public repo was
  running the old copy of the README.
- Added a live-demo link to the README. The Cloud Run deployment has been up and
  responding since 2026-05-28; nothing in the README pointed a reader at it.
- Corrected two stale claims that had drifted behind the actual repo state:
  - "Trained on three regions (SA1, NSW1, VIC1)" → all five NEM regions (SA1,
    NSW1, VIC1, QLD1, TAS1) have had trained models, cached price data, and
    backtest artifacts committed since 2026-05-28; the live demo already serves
    all five.
  - Committed-artifact size ("~2.6 MB") → actual tracked size is ~5.3 MB now
    that QLD1/TAS1 are included.
- Added a new honest-limitations finding, verified by reloading the committed
  QLD1 model + data and re-running the test-split evaluation directly (no
  retraining): QLD1's forecaster beats seasonal-naive on MAE (+18%), but the
  resulting *dispatch* policy earns slightly *less* revenue than the naive
  policy (59.8% vs 61.3% of perfect foresight). Forecast accuracy and downstream
  dispatch-decision quality aren't the same thing, and the project reports both.
- Independently re-verified the SA1 headline numbers and the per-region skill
  figures already in the README (MAE $61.52 vs naive $66.53, RMSE $164.50 vs
  $218.14; NSW1/SA1/VIC1 skill) by reloading the committed model weights and
  data and recomputing — all matched to two decimal places. No changes needed
  there; recorded here as the verification step, not a correction.
- Documented `api.py` (FastAPI service layer, added 2026-05-28) in the repo
  layout tree — it existed in the code but wasn't listed.
- Housekeeping: dropped an unintentional local-only edit to a tracked Chroma
  binary (`chroma_store/.../length.bin`, same size, no content change worth
  keeping).

## 2026-05-28 — initial portfolio build

- MILP dispatch optimiser (PuLP + HiGHS): charge/discharge against a price
  horizon with round-trip losses, SoC limits, an integer charge/discharge
  mutex, and a degradation cost.
- PyTorch multi-horizon LSTM price forecaster, evaluated leakage-free
  (scalers fit on the train split only) against a seasonal-naive baseline.
- Day-ahead receding-horizon backtest: perfect-foresight vs forecast-driven vs
  naive dispatch, settled at actual realised prices.
- LangGraph tool-calling agent (`recommend_dispatch`, `forecast_prices`,
  `search_market_docs`) backed by a Chroma RAG store over a hand-written
  NEM/BESS market-knowledge corpus.
- FastAPI service layer over the optimiser/forecaster/advisor.
- Data + model + Chroma artifacts trained and committed for NSW1 and VIC1
  alongside the original SA1 build.
- Free Google Cloud Run deploy path (build-from-source, scales to zero) added
  alongside the original AWS ECS Fargate path; demo deployed live to
  `australia-southeast1`.
