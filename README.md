# Dataset Reference Extraction with DocETL

A pipeline that extracts dataset references (accession numbers, repository
names, DOIs) from scientific papers using LLMs, and an evaluation harness
that scores results against the **DataRef-EXP** and **DataRef-REV** benchmarks
from [DataGatherer](https://github.com/VIDA-NYU/data-gatherer).

Each prediction is a row of the form:

```json
{"paper_id": "PMC1234567", "paper_doi": "10.x/yz",
 "dataset_identifier": "PXD009876", "repository": "PRIDE",
 "url": "https://www.ebi.ac.uk/pride/archive/projects/PXD009876"}
```

---

## Setup

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/Ezrill-Lin/DocETL-Pipeline-for-Data-Discovery-in-Scientific-Publications.git
cd DocETL-Pipeline-for-Data-Discovery-in-Scientific-Publications
uv sync
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY
```

> On Windows with OneDrive-synced paths, add `--link-mode=copy` to the `uv sync` call.

---

## Running the pipeline

Everything runs through a single entry point:

```bash
uv run python main.py [OPTIONS]
```

### Common examples

| Goal | Command |
|---|---|
| Full EXP benchmark (default) | `uv run python main.py` |
| Full REV benchmark | `uv run python main.py --benchmark rev` |
| Both benchmarks | `uv run python main.py --benchmarks exp,rev` |
| Skip DataGatherer (DocETL only) | `uv run python main.py --skip-datagatherer` |
| Reuse cached papers, skip fetch | `uv run python main.py --skip-fetch --skip-preprocess` |
| DocETL only, no DG, RTR only | `uv run python main.py --skip-fdr --skip-datagatherer` |
| Multiple models | `uv run python main.py --models gpt-4o-mini,gemini/gemini-2.0-flash` |

### All options

```
--benchmark {exp,rev}         Benchmark to run (default: exp)
--benchmarks BENCHMARKS       Comma-separated list, e.g. exp,rev (overrides --benchmark)
--format {xml,pdf}            EXP only: paper source format (default: xml)
--model MODEL                 LLM model, overrides config/settings.yaml
--models MODELS               Comma-separated models (overrides --model)
--raw-dir RAW_DIR             Override raw papers directory; disables auto-fetch
--groundtruth GROUNDTRUTH     Override benchmark ground-truth CSV
--benchmark-tag TAG           Tag for output paths (e.g. rev_100 for a 100-paper smoke test)
--skip-preprocess             Reuse existing papers.json
--skip-fetch                  Skip downloading benchmark papers
--skip-rtr                    Skip RTR pipeline
--skip-fdr                    Skip FDR pipeline
--skip-datagatherer           Skip DataGatherer baseline
--parallel                    Run all extraction methods in parallel
--max-threads N               Max concurrent LLM threads per DocETL pipeline (default: 10)
```

### What the pipeline does

1. **Fetch** — Downloads paper XMLs from PMC and benchmark ground truth from Zenodo.
2. **Preprocess** — Parses JATS XML into structured records with `candidate_passages` (for RTR) and `full_text` (for FDR).
3. **Extract** — Runs up to four methods:
   - **DocETL-RTR**: LLM reads candidate passages only (fast, lower recall).
   - **DocETL-FDR**: LLM reads the full document (higher recall, higher cost).
   - **DG-RTR**: DataGatherer with section-level retrieval.
   - **DG-FDR**: DataGatherer with full-document read.
4. **Evaluate** — Computes pair-level and triple-level precision / recall / F1 (micro and macro).
5. **Report** — Writes `outputs/report.md` with a unified comparison table.

---

## Outputs

| Path | Contents |
|---|---|
| `data/predictions/<benchmark>/<model>/rtr.jsonl` | DocETL-RTR predictions |
| `data/predictions/<benchmark>/<model>/fdr.jsonl` | DocETL-FDR predictions |
| `data/predictions/<benchmark>/<model>/datagatherer_rtr.jsonl` | DG-RTR predictions |
| `data/predictions/<benchmark>/<model>/datagatherer_fdr.jsonl` | DG-FDR predictions |
| `outputs/metrics/<benchmark>_<model>_rtr.json` | Full metrics for DocETL-RTR |
| `outputs/metrics/<benchmark>_<model>_fdr.json` | Full metrics for DocETL-FDR |
| `outputs/metrics/<benchmark>_<model>_datagatherer_rtr.json` | Full metrics for DG-RTR |
| `outputs/metrics/<benchmark>_<model>_datagatherer_fdr.json` | Full metrics for DG-FDR |
| `outputs/report.md` | Unified Markdown comparison report |

---

## Reproducibility benchmark (EXP)

Because `gpt-4o-mini` is non-deterministic, we provide a script that runs
DocETL RTR + FDR on DataRef-EXP N times and reports mean ± std:

```bash
uv run python scripts/benchmark_docetl_exp.py --runs 20 --max-threads 21
```

Results are saved to `outputs/docetl_exp_benchmark.json`.

---

## Tests

```bash
uv run pytest tests/ -q
```

Unit tests cover URL construction, identifier normalisation, paper-key
matching, and metric computation.

---

## Project layout

```
config/                 repositories.yaml, settings.yaml
pipelines/              DocETL pipeline YAMLs (RTR + FDR)
src/
  preprocess/           JATS XML / PDF → structured paper records
  extraction/           DocETL runner, URL builder, output normalisation
  evaluation/           Ground-truth loader, pair/triple matcher, metrics
  baselines/            DataGatherer wrapper
  reporting/            Markdown report generator
scripts/                Utility scripts (benchmark, token estimation, etc.)
tests/                  pytest unit tests
data/
  raw/                  Downloaded paper XMLs
  processed/            Preprocessed paper JSON
  benchmark/            Ground-truth CSVs
  predictions/          Per-method JSONL output
outputs/                Metrics JSON + report.md
```

---

## Known limitations

- **DG-RTR on REV**: A network outage during our REV run left ~45 % of papers with empty output; those metrics are underestimated. Re-run with a stable connection for a fair comparison.
- **Cost tracking**: DocETL reports a framework-level total. Per-paper cost is estimated from character counts. DataGatherer only exposes input tokens (output estimated at 10 %).
- **URL templates**: `config/repositories.yaml` covers the repositories in the benchmark. Unknown repositories get an empty URL field — URLs are never fabricated.
- **PDF parsing**: Heading detection on PDFs is heuristic. JATS XML produces better structure.

---

## API keys

Copy `.env.example` to `.env` and fill in your key:

```
OPENAI_API_KEY=sk-...
```

Never commit a populated `.env`. Only `.env.example` is tracked.

