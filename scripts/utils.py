"""
Utility Functions for DocETL Dataset Extraction Pipeline

This script provides helper utilities for input preparation, result analysis, and data export.
"""

import json
import os
from pathlib import Path
from typing import List, Dict


def create_papers_input(papers_directory: str = "papers", output_file: str = "papers_input.json") -> None:
    """
    Scan a directory of PDF/HTML papers and create the input JSON file for DocETL.
    
    Args:
        papers_directory: Path to directory containing paper files
        output_file: Path to output JSON file
    """
    papers_dir = Path(papers_directory)
    
    if not papers_dir.exists():
        print(f"Creating directory: {papers_directory}")
        papers_dir.mkdir(parents=True, exist_ok=True)
        print(f"Please add your PDF files to the '{papers_directory}' directory")
        return
    
    papers = []
    
    # Scan for PDF files
    pdf_files = list(papers_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{papers_directory}' directory")
        print("Please add PDF files and run again")
        return
    
    for idx, pdf_file in enumerate(pdf_files, start=1):
        paper_entry = {
            "paper_id": f"paper_{idx:03d}",
            "title": pdf_file.stem.replace("_", " ").replace("-", " "),
            "pdf_path": str(pdf_file)
        }
        papers.append(paper_entry)
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    print(f"Created {output_file} with {len(papers)} papers")
    print(f"Papers found: {[p['title'] for p in papers]}")


def analyze_output(output_file: str = "dataset_references_output.json") -> None:
    """
    Analyze and display statistics from the pipeline output.
    
    Args:
        output_file: Path to output JSON file
    """
    if not os.path.exists(output_file):
        print(f"Output file not found: {output_file}")
        return
    
    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("\n" + "="*60)
    print("DATASET EXTRACTION RESULTS")
    print("="*60)
    
    print(f"\nTotal unique datasets found: {len(results)}")
    
    # Sort by number of papers using the dataset
    results_sorted = sorted(results, key=lambda x: x.get('num_papers', 0), reverse=True)
    
    print("\nTop datasets by usage:")
    for i, dataset in enumerate(results_sorted[:10], 1):
        print(f"\n{i}. {dataset.get('dataset_identifier', 'Unknown')}")
        print(f"   Repository: {dataset.get('repository', 'N/A')}")
        print(f"   URL: {dataset.get('url', 'N/A')}")
        print(f"   Used in {dataset.get('num_papers', 0)} paper(s)")
    
    # Statistics
    total_papers = sum(d.get('num_papers', 0) for d in results)
    datasets_with_url = sum(1 for d in results if d.get('url'))
    datasets_with_repo = sum(1 for d in results if d.get('repository'))
    
    print("\n" + "-"*60)
    print("STATISTICS:")
    print(f"  Total dataset mentions across papers: {total_papers}")
    print(f"  Datasets with URL: {datasets_with_url} ({datasets_with_url/len(results)*100:.1f}%)")
    print(f"  Datasets with repository info: {datasets_with_repo} ({datasets_with_repo/len(results)*100:.1f}%)")
    print("="*60)


def export_to_csv(
    output_file: str = "dataset_references_output.json",
    csv_file: str = "datasets.csv"
) -> None:
    """
    Export the results to a CSV file for easier analysis.
    
    Args:
        output_file: Path to JSON output file
        csv_file: Path to output CSV file
    """
    import csv
    
    if not os.path.exists(output_file):
        print(f"Output file not found: {output_file}")
        return
    
    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset_identifier', 'repository', 'url', 'num_papers'
        ])
        writer.writeheader()
        
        for dataset in results:
            writer.writerow({
                'dataset_identifier': dataset.get('dataset_identifier', ''),
                'repository': dataset.get('repository', ''),
                'url': dataset.get('url', ''),
                'num_papers': dataset.get('num_papers', 0)
            })
    
    print(f"Exported results to {csv_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("DocETL Pipeline Utility Functions")
        print("\nUsage:")
        print("  python utils.py prepare    - Create input JSON from papers directory")
        print("  python utils.py analyze    - Analyze pipeline output")
        print("  python utils.py export     - Export results to CSV")
        print("\nNote: To run pipelines, use pipelines/main.py instead")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "prepare":
        create_papers_input()
    elif command == "analyze":
        analyze_output()
    elif command == "export":
        export_to_csv()
    else:
        print(f"Unknown command: {command}")
        print("Valid commands: prepare, analyze, export")
