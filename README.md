# DocETL Pipeline for Dataset Reference Extraction from Scientific Publications

An automated pipeline using DocETL and large language models to extract structured dataset references from scientific papers (PDF, HTML, XML formats).

## 📁 Project Structure

```
DocETL-Pipeline-for-Data-Discovery-in-Scientific-Publications/
├── pipelines/                      # 🎯 Main pipeline directory
│   ├── main.py                    # ⭐ Main entry point - run this!
│   ├── pipeline_pdf.yaml          # PDF/HTML papers pipeline config
│   └── pipeline_xml.yaml          # XML papers pipeline config
│
├── data/                          # All data files
│   ├── input/                     # Input papers
│   │   ├── pdf/                   # PDF papers and papers_input.json
│   │   ├── xml/                   # XML papers and xml_papers_input.json
│   │   └── html/                  # HTML papers (future)
│   ├── output/                    # Pipeline results
│   │   ├── pdf_results/          # PDF pipeline outputs
│   │   └── xml_results/          # XML pipeline outputs
│   └── groundtruth/              # Evaluation datasets
│       ├── EXP_groundtruth.csv
│       └── Full_REV_dataset_citation_records_Table.parquet
│
├── scripts/                       # 🛠️ Utility scripts (not for direct pipeline execution)
│   ├── etl.py                    # Analysis and export utilities
│   └── download_papers.py        # Paper download utilities
│
├── docs/                         # Documentation
│   ├── README.md                 # Main project documentation
│   ├── PARSING_INFO.md           # Parsing details
│   └── xml_pipeline_README.md    # XML pipeline guide
│
├── intermediate_results/          # Temporary processing files
│   ├── pdf_pipeline/             # PDF pipeline intermediates
│   └── xml_pipeline/             # XML pipeline intermediates
│
├── docetl/                       # Virtual environment (not in git)
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.12+** installed
2. **OpenAI API Key** (set as environment variable)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd DocETL-Pipeline-for-Data-Discovery-in-Scientific-Publications

# Create and activate virtual environment
python -m venv docetl
.\docetl\Scripts\Activate.ps1  # Windows PowerShell
# or: source docetl/bin/activate  # Linux/Mac

# Install dependencies
pip install docetl[parsing] litellm

# Set your OpenAI API key
$env:OPENAI_API_KEY = "your-api-key-here"  # Windows PowerShell
# or: export OPENAI_API_KEY="your-api-key-here"  # Linux/Mac
```

### Running the Pipelines

The easiest way to run the pipelines is using the **main.py** script:

```bash
# Navigate to pipelines directory
cd pipelines

# Run PDF pipeline
python main.py pdf

# Run XML pipeline
python main.py xml

# Run both pipelines
python main.py all

# Run with optimization
python main.py pdf --optimize
```

**Alternative: Direct DocETL commands**

```bash
# Navigate to pipelines directory
cd pipelines

# PDF Pipeline
docetl run pipeline_pdf.yaml
# Output: ../data/output/pdf_results/dataset_references_output.json

# XML Pipeline
docetl run pipeline_xml.yaml
# Output: ../data/output/xml_results/xml_dataset_references_output.json
```

## 📊 Output Format

Both pipelines produce JSON output with structured dataset references:

```json
[
  {
    "dataset_identifier": "DataRef-EXP",
    "repository": "PubMed Central",
    "url": "https://pmc.ncbi.nlm.nih.gov/",
    "num_papers": 1,
    "paper_paths": [
      "data/input/pdf/2025.sdp-1.10.pdf"
    ]
  }
]
```

## 🔧 Pipeline Architecture

Each pipeline consists of 6 operations:

1. **extract_datasets** - LLM extracts dataset mentions from paper text
2. **unnest_datasets** - Separates each dataset into individual records
3. **flatten_dataset_fields** - Extracts nested fields to top level
4. **filter_valid_datasets** - Removes invalid/placeholder entries
5. **resolve_datasets** - Groups similar dataset names together
6. **aggregate_dataset_usage** - Consolidates info and counts paper usage

## 📝 Adding New Papers

### PDF Papers

1. Place PDF files in `data/input/pdf/`
2. Update `data/input/pdf/papers_input.json`:

```json
[
  {
    "paper_id": "unique_id",
    "title": "Paper Title",
    "pdf_path": "data/input/pdf/filename.pdf"
  }
]
```

3. Run `docetl run pipeline_pdf.yaml` from `pipelines/` directory

### XML Papers

1. Place XML files in `data/input/xml/`
2. Regenerate input file:

```bash
python -c "import json; from pathlib import Path; papers = [{'paper_id': f.stem, 'title': f.stem, 'xml_path': f'data/input/xml/{f.name}'} for f in Path('data/input/xml').glob('*.xml')]; json.dump(papers, open('data/input/xml/xml_papers_input.json', 'w'), indent=2)"
```

3. Run `docetl run pipeline_xml.yaml` from `pipelines/` directory

## 📚 Documentation

- **[docs/README.md](docs/README.md)** - Detailed project documentation
- **[docs/PARSING_INFO.md](docs/PARSING_INFO.md)** - Information about parsing libraries
- **[docs/xml_pipeline_README.md](docs/xml_pipeline_README.md)** - XML pipeline specifics

## 🧪 Evaluation

Ground truth datasets for evaluation are stored in `data/groundtruth/`:

- `EXP_groundtruth.csv` - Manually curated expert dataset (21 papers, 48 references)
- `Full_REV_dataset_citation_records_Table.parquet` - Large-scale reverse-engineered dataset (244,847 papers, 397,263 references)

## 🛠️ Utility Scripts (scripts/ directory)

The `scripts/` folder contains helper utilities for supporting tasks. These are NOT used for running the main pipeline (use `pipelines/main.py` for that).

### scripts/etl.py - Analysis & Export Utilities

Provides additional functionality for pipeline outputs:

```bash
# Analyze pipeline results
python scripts/etl.py analyze

# Export results to CSV
python scripts/etl.py export

# Run analysis on specific file
python scripts/etl.py analyze --file data/output/pdf_results/dataset_references_output.json
```

Commands:
- `prepare` - Scan papers directory and create input.json
- `run` - Execute the pipeline
- `analyze` - Generate statistics and visualizations
- `export` - Convert JSON output to CSV

### scripts/download_papers.py - Paper Download Utility

Downloads papers from PubMed Central using E-utilities API.

```bash
python scripts/download_papers.py
```

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
OPENAI_API_KEY=your-api-key-here
```

### Pipeline Customization

Edit YAML files in `pipelines/` to:
- Change LLM models (default: gpt-4o-mini)
- Modify extraction prompts
- Adjust validation rules
- Configure output schemas

## 📦 Dependencies

Core dependencies:
- `docetl[parsing]` - DocETL with parsing extras
- `litellm` - LLM provider interface
- `PyMuPDF (fitz)` - PDF text extraction
- `openai` - OpenAI API client

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- Built with [DocETL](https://github.com/ucbepic/docetl)
- Uses OpenAI's GPT models for extraction
- Based on research from "Data Gatherer: LLM-Powered Dataset Reference Extraction from Scientific Literature"

## 📞 Contact

[Your Contact Information]


## Project Structure

```
.
├── pipeline.yaml           # DocETL pipeline configuration
├── papers_input.json       # Input: List of papers to process
├── papers/                 # Directory containing PDF papers
│   ├── paper_001.pdf
│   ├── paper_002.pdf
│   └── ...
├── dataset_references_output.json  # Output: Extracted dataset references
├── intermediate_results/   # Intermediate processing results
└── .env                    # API keys (not committed to git)
```

## Usage

### 1. Prepare Your Input Data

Create a `papers_input.json` file with references to your PDF papers:

```json
[
  {
    "paper_id": "paper_001",
    "title": "Your Paper Title",
    "pdf_path": "papers/paper_001.pdf"
  }
]
```

Place your PDF files in the `papers/` directory.

### 2. Run the Pipeline

```bash
docetl run pipeline.yaml
```

### 3. Check the Output

The pipeline will create:
- `dataset_references_output.json`: Final dataset references with usage statistics
- `intermediate_results/`: Step-by-step processing results for debugging

## Pipeline Steps

1. **PDF Parsing**: Extracts text content from PDF files using OCR
2. **Dataset Extraction**: Uses LLM to identify dataset references in each paper
3. **Unnesting**: Flattens the dataset list for individual processing
4. **Filtering**: Removes invalid or generic dataset entries
5. **Resolution**: Standardizes similar dataset names (e.g., "ImageNet-1K" → "ImageNet")
6. **Aggregation**: Consolidates information and tracks paper usage

## Output Format

```json
[
  {
    "dataset_identifier": "ImageNet",
    "repository": "image-net.org",
    "url": "https://image-net.org/",
    "num_papers": 5,
    "paper_paths": ["papers/paper_001.pdf", "papers/paper_003.pdf", ...]
  }
]
```

## Customization

### Using HTML Instead of PDF

To process HTML papers, modify the `pipeline.yaml` parsing section:

```yaml
datasets:
  scientific_papers:
    type: file
    source: local
    path: "papers_input.json"
    parsing:
      - input_key: html_path
        function: txt_to_string  # or create custom HTML parser
        output_key: paper_content
```

### Processing a Sample Dataset

To test on a smaller sample, add `sample: 10` to the map operation:

```yaml
operations:
  - name: extract_datasets
    type: map
    sample: 10  # Process only 10 papers
    prompt: |
      ...
```

### Using Different LLM Models

Change the model in `pipeline.yaml`:

```yaml
default_model: gpt-4o  # More powerful but more expensive
# or
default_model: gemini/gemini-2.0-flash  # Cheaper alternative
```

### Adjusting Dataset Resolution Threshold

Modify the `blocking_threshold` in the resolve operation (0.0 to 1.0):

```yaml
- name: resolve_datasets
  type: resolve
  blocking_threshold: 0.7  # Lower = more aggressive matching
```

## Cost Estimation

Using `gpt-4o-mini`:
- Small corpus (10 papers): ~$0.10 - $0.50
- Medium corpus (100 papers): ~$1.00 - $5.00
- Large corpus (1000 papers): ~$10.00 - $50.00

Costs depend on paper length and number of datasets found.

## Troubleshooting

### OCR Issues
If PDF parsing fails:
- Ensure papers are not encrypted or password-protected
- For image-heavy PDFs, OCR may be slow (5-30 seconds per page)
- Consider using Azure Document Intelligence for better accuracy:
  ```yaml
  parsing:
    - input_key: pdf_path
      function: azure_di_read
      output_key: paper_content
  ```

### Memory Issues
For large corpora:
- Process in batches by splitting `papers_input.json`
- Reduce `blocking_threshold` to avoid excessive comparisons
- Use a simpler model like `gpt-4o-mini`

### API Rate Limits
DocETL automatically caches LLM calls. If you hit rate limits:
- Add delays between requests
- Use a higher-tier API plan
- Process smaller batches

## Python API Alternative

You can also run the pipeline using Python:

```python
from docetl.api import Pipeline

pipeline = Pipeline.from_yaml("pipeline.yaml")
results = pipeline.run()
print(results)
```

## Contributing

To extend this pipeline:
1. Add custom parsing tools for specific paper formats
2. Implement additional validation rules
3. Add enrichment steps (e.g., fetch dataset metadata from DOIs)

## References

- [DocETL Documentation](https://ucbepic.github.io/docetl/)
- [DocETL GitHub](https://github.com/ucbepic/docetl)
- [Quick Start Tutorial](https://ucbepic.github.io/docetl/tutorial/)
