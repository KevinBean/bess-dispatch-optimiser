# Convenience targets. On Windows, run these from Git Bash or WSL, or copy the
# underlying commands. PY points at the project venv interpreter.
PY := .venv/Scripts/python.exe
REGION ?= SA1

.PHONY: artifacts data train backtest ingest test docker run-local

## Build everything the demo image bakes (data + model + chroma store).
artifacts: data train ingest backtest

data:
	$(PY) scripts/fetch_data.py --region $(REGION) --start 2024-01 --end 2024-12

train:
	$(PY) scripts/train_forecast.py --region $(REGION) --epochs 40

backtest:
	$(PY) scripts/run_backtest.py --region $(REGION) --test-start 2024-11-15

ingest:
	$(PY) -c "from src.bess.rag import RagStore; print('chunks:', RagStore().ingest_dir())"

test:
	$(PY) -m pytest -q

docker:
	docker build -t bess-optimiser:latest .

run-local:
	docker run --rm -p 8501:8501 -e OPENAI_API_KEY=$$OPENAI_API_KEY bess-optimiser:latest
