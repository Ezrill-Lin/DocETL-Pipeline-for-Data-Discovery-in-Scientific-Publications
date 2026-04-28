# DocETL Pipeline for Scientific Paper Dataset Extraction

This project runs two DocETL pipelines over scientific papers:

- a PDF pipeline using `PyMuPDF`
- an XML pipeline using section-aware XML parsing

Both pipelines extract dataset references and write normalized JSON outputs.

## Project Layout

```text
DocETL-Pipeline-for-Data-Discovery-in-Scientific-Publications/
├── data/
│   ├── groundtruth/
│   ├── input/
│   │   ├── html/
│   │   ├── pdf/
│   │   └── xml/
│   └── output/
├── docetl_venv/
├── intermediate_results/
├── pipelines/
│   ├── pipeline_pdf.yaml
│   └── pipeline_xml.yaml
├── scripts/
│   ├── download_papers.py
│   ├── download_papers_pdf.py
│   ├── eval.py
│   └── utils.py
├── .env.example
├── .gitignore
└── main.py
```

## Requirements

- Python 3.12+
- A working project virtual environment at `docetl_venv/`
- `OPENAI_API_KEY`

`main.py` uses the project-local executable at `docetl_venv/Scripts/docetl.exe`. It does not depend on a globally installed `docetl`.

## Environment Setup

Create `.env` from `.env.example` or set variables in your shell.

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

Optional:

```powershell
$env:DOCETL_MAX_THREADS = "2"
```

Use `DOCETL_MAX_THREADS` if your API tier hits rate limits under the default DocETL concurrency.

## Running the Pipelines

Run commands from the project root:

```powershell
python main.py pdf
python main.py xml
python main.py all
python main.py help
```

What `main.py` does:

- loads `.env` if present
- regenerates `data/input/pdf/papers_input.json`
- regenerates `data/input/xml/xml_papers_input.json`
- runs the selected pipeline from `pipelines/`

## Inputs

### PDF pipeline

Place `.pdf` files in `data/input/pdf/`.

The manifest `data/input/pdf/papers_input.json` is generated automatically by `main.py`.

### XML pipeline

Place `.xml` files in `data/input/xml/`.

The manifest `data/input/xml/xml_papers_input.json` is generated automatically by `main.py`.

## Outputs

Pipeline outputs:

- PDF: `data/output/pdf_results/dataset_references_output.json`
- XML: `data/output/xml_results/xml_dataset_references_output.json`

Intermediate caches:

- PDF: `intermediate_results/pdf_pipeline/`
- XML: `intermediate_results/xml_pipeline/`

## Download Scripts

The downloader utilities are separate from the DocETL runs:

- `scripts/download_papers.py` for general paper download helpers
- `scripts/download_papers_pdf.py` for browser-mediated PMC PDF downloads

These scripts prepare inputs for the pipelines, but `main.py` is the entrypoint for running DocETL.

## Notes

- The old `--optimize` flag is not part of the current `main.py` interface.
- The project currently works with the local `docetl_venv` executable.
- If you see API rate-limit retries, lower `DOCETL_MAX_THREADS`.
