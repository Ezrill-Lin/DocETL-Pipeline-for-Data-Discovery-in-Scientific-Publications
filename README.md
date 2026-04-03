# DocETL Pipeline for Dataset Discovery in Scientific Publications

This project uses DocETL to build an ETL pipeline that extracts dataset references from scientific papers (PDF or HTML format) and outputs structured dataset information.

## Features

- **Automatic Dataset Extraction**: Uses LLMs to identify dataset references in scientific papers
- **Multi-format Support**: Handles PDF files (with OCR support) and can be extended for HTML
- **Dataset Resolution**: Automatically identifies and merges similar dataset names
- **Structured Output**: Produces clean JSON output with dataset identifiers, repositories, and URLs
- **Paper Tracking**: Maintains links between datasets and the papers that use them

## Prerequisites

- Python 3.10 or later
- OpenAI API key (or other LLM provider)
- DocETL with parsing extras installed

## Installation

1. **Create and activate the virtual environment** (already done):
   ```bash
   .\docetl\Scripts\Activate.ps1
   ```

2. **Install DocETL with parsing extras**:
   ```bash
   pip install docetl[parsing]
   ```

3. **Set up your OpenAI API key**:
   
   Create a `.env` file in the project directory:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   
   Or set it as an environment variable:
   ```bash
   $env:OPENAI_API_KEY="your_api_key_here"
   ```

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
