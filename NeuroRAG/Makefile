# ---------------------------------------------------------------------------
# NeuroRAG - common tasks
#
#   make setup      install dependencies
#   make corpus     fetch a fresh literature snapshot from Europe PMC + arXiv
#   make golden     (re)build and validate the golden evaluation set
#   make index      build the retrieval index
#   make calibrate  learn the abstention threshold from the golden set
#   make eval       full evaluation + comparison + weight sweep -> docs/EVALUATION.md
#   make demo       launch the Streamlit UI
#   make ask        one-shot question from the CLI
#   make test       run the test suite
#   make offline    build an index with no ML dependencies at all
#   make clean      remove generated artefacts (keeps the committed corpus)
# ---------------------------------------------------------------------------

PYTHON ?= python3
Q      ?= "Why is subject-level cross-validation important for MRI classification?"

.PHONY: help setup corpus golden index calibrate eval demo ask test offline clean verify

help:
	@sed -n '2,15p' Makefile

setup:
	$(PYTHON) -m pip install -r requirements.txt

corpus:
	$(PYTHON) scripts/build_corpus.py

golden:
	$(PYTHON) scripts/build_golden.py

index:
	$(PYTHON) scripts/build_index.py

calibrate:
	$(PYTHON) scripts/evaluate.py --calibrate --retrieval-only --report /dev/null

eval:
	$(PYTHON) scripts/evaluate.py --compare --sweep-weights --show-worst 5

demo:
	streamlit run app.py

ask:
	$(PYTHON) scripts/ask.py $(Q) --explain

test:
	$(PYTHON) -m pytest tests/ -v

# Builds and evaluates with the offline lexical encoder and extractive backend.
# Proves the repository runs end to end on a bare Python install.
offline:
	$(PYTHON) scripts/build_index.py --lexical --no-rerank
	$(PYTHON) scripts/evaluate.py --set llm=extractive --show-worst 3

verify:
	$(PYTHON) scripts/build_corpus.py --verify
	$(PYTHON) scripts/build_golden.py

clean:
	rm -rf data/index data/index_ablation data/cache reports .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "Removed generated artefacts. The committed corpus and golden set are kept."
