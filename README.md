# Dataset Reference Extraction with DocETL

A DocETL-based pipeline that extracts dataset references (accession numbers,
repository names, DOIs) from scientific papers, and an evaluation harness
that scores it against the **DataRef-EXP** / **DataRef-REV** benchmarks
released alongside [DataGatherer](https://github.com/VIDA-NYU/data-gatherer).

The pipeline emits one row per (paper, dataset reference):

```json
{"paper_id": "PMC1234567", "paper_doi": "10.x/yz",
 "dataset_identifier": "PXD009876", "repository": "PRIDE",
 "url": "https://www.ebi.ac.uk/pride/archive/projects/PXD009876"}
```

## Why this design

- **Hybrid retrieval + LLM extraction.** Pure heading-based extraction
  misses references buried in inline text and figure captions; pure
  LLM-on-full-text is wasteful. We pre-select candidate passages by
  heading match *and* by regex over repository names and accession-shaped
  tokens, then ask the LLM to extract from the (much smaller) candidate
  text.
- **Deterministic URL construction.** The LLM extracts identifiers and
  repository names but never URLs. URLs are templated in Python
  (`src/extraction/url_builder.py`) from `config/repositories.yaml`. If
  the repository is unknown, the URL field stays empty rather than being
  fabricated.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in OPENAI_API_KEY (or ANTHROPIC_API_KEY) and DOCETL_MODEL
```

`DOCETL_MODEL` selects the LiteLLM-compatible model (e.g. `gpt-4o-mini`,
`claude-haiku-4-5-20251001`).

## Project layout

```
config/                 repositories + settings
pipelines/              DocETL pipeline YAML
src/preprocess/         HTML / PDF → structured paper records
src/extraction/         DocETL runner + URL builder + output normalization
src/evaluation/         ground truth loader + matcher + metrics
src/baselines/          DataGatherer wrapper (best-effort)
src/reporting/          Markdown report generator
scripts/                CLI entry points
tests/                  pytest unit tests
data/                   raw papers, processed JSON, benchmark, predictions
outputs/                metrics + final report
```

## How to run

### 1. Download benchmark ground truth

```bash
python scripts/download_benchmarks.py
```

Pulls every file from Zenodo record `15549086` into `data/benchmark/`. If
network access is blocked, follow the manual instructions printed by the
script and place the CSV ground-truth file there yourself.

### 2. Prepare paper inputs

Drop HTML, JATS XML, or PDF files into `data/raw/`, then:

```bash
python scripts/prepare_inputs.py --input data/raw --output data/processed/papers.json
```

This produces a JSON array of paper records, each with an extracted section
tree and a `candidate_passages` field that the LLM will read.

### 3. Run the DocETL extraction pipeline

```bash
python scripts/run_pipeline.py \
  --input data/processed/papers.json \
  --output data/predictions/docetl_predictions.jsonl
```

The pipeline writes raw DocETL output to
`data/predictions/docetl_predictions.raw.json`, then flattens / URL-builds
into `docetl_predictions.jsonl`. Cost / token estimates land in
`outputs/cost_docetl.json`.

### 4. Evaluate

```bash
python scripts/evaluate_docetl.py \
  --predictions data/predictions/docetl_predictions.jsonl \
  --groundtruth data/benchmark/EXP_groundtruth.csv \
  --output outputs/metrics_docetl.json
```

Computes pair-level and repository-aware (triple-level) precision /
recall / F1, plus per-paper macro averages, coverage, and failure
categories.

### 5. (Optional) Run DataGatherer baseline

```bash
pip install git+https://github.com/VIDA-NYU/data-gatherer
python -c "from src.baselines.run_datagatherer import run_datagatherer; \
           run_datagatherer('data/processed/papers.json', \
                            'data/predictions/datagatherer_predictions.jsonl')"
```

`src/baselines/run_datagatherer.py` probes a few known entry points; if
your installed version exposes a different API, edit
`_build_extractor()`. If DataGatherer is unavailable the wrapper writes
an empty predictions file and the evaluation continues without it.

### 6. End-to-end

```bash
python scripts/run_all.py --groundtruth data/benchmark/EXP_groundtruth.csv
```

Each stage is best-effort — missing inputs or failed sub-runs are logged
and the next stage continues. The final markdown report is written to
`outputs/report.md`.

## Tests

```bash
pytest tests/ -q
```

21 unit tests cover URL construction, identifier and repository
normalization, paper-key matching, and metric computation.

## Outputs

- `data/predictions/docetl_predictions.jsonl` — one row per dataset reference
- `outputs/cost_docetl.json` — token/cost estimate for the DocETL run
- `outputs/metrics_docetl.json` — full metrics + failure categories
- `outputs/metrics_datagatherer.json` — same, if the baseline ran
- `outputs/report.md` — narrative comparison

## Known limitations

- **Cost tracking is approximate.** DocETL reports a total cost via
  `load_run_save()`; per-paper attribution is estimated from input
  character counts.
- **PDF parsing.** Heading detection on PDFs is heuristic. JATS XML or
  PMC HTML produces much better structure than scanned PDFs.
- **DataGatherer API drift.** The baseline wrapper probes plausible
  entry points; for older / forked versions you may need to edit
  `_build_extractor` or run DataGatherer manually and drop its
  predictions into `data/predictions/datagatherer_predictions.jsonl`.
- **URL templates.** `config/repositories.yaml` covers the repositories
  in the project spec. For others, the URL field is left empty; we never
  fabricate URLs.
- **Ground truth column names.** The loader tries common column names
  (`accession`, `dataset_identifier`, `database`, `repository`, …). If
  your benchmark CSV uses different headers, edit
  `src/evaluation/load_groundtruth.py:COLUMN_CANDIDATES`.

## API keys and secrets

`.env` is read at runtime. Never commit a populated `.env`. The
`.env.example` file is the only one tracked in version control.
