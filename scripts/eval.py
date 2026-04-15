"""
Evaluation Script for DocETL Pipeline Results

Compares pipeline-extracted dataset references against ground truth data
and calculates precision, recall, and F1 scores.

Usage:
    python scripts/eval.py <pipeline_results.json> <groundtruth.csv>
    python scripts/eval.py --xml  # Evaluate XML pipeline against EXP ground truth
    python scripts/eval.py --pdf  # Evaluate PDF pipeline against ground truth
"""

import json
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import sys


class PipelineEvaluator:
    """Evaluates pipeline results against ground truth."""
    
    def __init__(self, results_file: str, groundtruth_file: str):
        """
        Initialize evaluator.
        
        Args:
            results_file: Path to pipeline results JSON
            groundtruth_file: Path to ground truth CSV or Parquet
        """
        self.results_file = results_file
        self.groundtruth_file = groundtruth_file
        
        # Load data
        self.results = self._load_results()
        self.groundtruth = self._load_groundtruth()
        
        # Extract paper-dataset mappings
        self.predicted_mappings = self._extract_predicted_mappings()
        self.true_mappings = self._extract_true_mappings()
    
    def _load_results(self) -> List[Dict]:
        """Load pipeline results from JSON."""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_groundtruth(self) -> pd.DataFrame:
        """Load ground truth from CSV or Parquet."""
        file_path = Path(self.groundtruth_file)
        
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _extract_pmc_id(self, text: str) -> str:
        """
        Extract PMC ID from various formats.
        
        Args:
            text: Text containing PMC ID (URL, file path, etc.)
            
        Returns:
            PMC ID (e.g., 'PMC11252349') or empty string if not found
        """
        # Match PMC followed by digits
        match = re.search(r'PMC\d+', text, re.IGNORECASE)
        return match.group(0).upper() if match else ''
    
    def _normalize_identifier(self, identifier: str) -> str:
        """
        Normalize dataset identifier for comparison.
        
        Args:
            identifier: Dataset identifier
            
        Returns:
            Normalized identifier (uppercase, stripped)
        """
        if not identifier or identifier == 'N/A':
            return ''
        return str(identifier).strip().upper()
    
    def _extract_predicted_mappings(self) -> Dict[str, Set[str]]:
        """
        Extract paper-to-datasets mappings from pipeline results.
        
        Returns:
            Dict mapping PMC ID to set of dataset identifiers
        """
        mappings = defaultdict(set)
        
        for item in self.results:
            dataset_id = self._normalize_identifier(item.get('dataset_identifier', ''))
            
            if not dataset_id:
                continue
            
            # Extract PMC IDs from paper paths
            for paper_path in item.get('paper_paths', []):
                pmc_id = self._extract_pmc_id(paper_path)
                if pmc_id:
                    mappings[pmc_id].add(dataset_id)
        
        return dict(mappings)
    
    def _extract_true_mappings(self) -> Dict[str, Set[str]]:
        """
        Extract paper-to-datasets mappings from ground truth.
        
        Returns:
            Dict mapping PMC ID to set of dataset identifiers
        """
        mappings = defaultdict(set)
        
        # Handle different column names
        paper_col = 'citing_publication_link'
        id_col = 'identifier'
        
        for _, row in self.groundtruth.iterrows():
            # Extract PMC ID from paper link
            paper_link = str(row.get(paper_col, ''))
            pmc_id = self._extract_pmc_id(paper_link)
            
            # Get dataset identifier
            dataset_id = self._normalize_identifier(row.get(id_col, ''))
            
            if pmc_id and dataset_id:
                mappings[pmc_id].add(dataset_id)
        
        return dict(mappings)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Returns:
            Dict with precision, recall, f1, and counts
        """
        # Get all PMC IDs that appear in both predicted and ground truth
        common_pmcs = set(self.predicted_mappings.keys()) & set(self.true_mappings.keys())
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Calculate TP, FP, FN for common papers
        for pmc_id in common_pmcs:
            predicted = self.predicted_mappings[pmc_id]
            true = self.true_mappings[pmc_id]
            
            # True positives: correctly predicted datasets
            tp = len(predicted & true)
            # False positives: predicted but not in ground truth
            fp = len(predicted - true)
            # False negatives: in ground truth but not predicted
            fn = len(true - predicted)
            
            true_positives += tp
            false_positives += fp
            false_negatives += fn
        
        # Papers only in predictions (all are false positives)
        for pmc_id in set(self.predicted_mappings.keys()) - common_pmcs:
            false_positives += len(self.predicted_mappings[pmc_id])
        
        # Papers only in ground truth (all are false negatives)
        for pmc_id in set(self.true_mappings.keys()) - common_pmcs:
            false_negatives += len(self.true_mappings[pmc_id])
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'common_papers': len(common_pmcs),
            'predicted_papers': len(self.predicted_mappings),
            'groundtruth_papers': len(self.true_mappings)
        }
    
    def get_detailed_results(self) -> Dict:
        """
        Get detailed evaluation results by paper.
        
        Returns:
            Dict with per-paper breakdowns
        """
        results = {}
        common_pmcs = set(self.predicted_mappings.keys()) & set(self.true_mappings.keys())
        
        for pmc_id in sorted(common_pmcs):
            predicted = self.predicted_mappings[pmc_id]
            true = self.true_mappings[pmc_id]
            
            results[pmc_id] = {
                'true_positives': list(predicted & true),
                'false_positives': list(predicted - true),
                'false_negatives': list(true - predicted),
                'precision': len(predicted & true) / len(predicted) if predicted else 0,
                'recall': len(predicted & true) / len(true) if true else 0
            }
        
        return results
    
    def print_evaluation_report(self):
        """Print detailed evaluation report."""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*80)
        print("PIPELINE EVALUATION REPORT")
        print("="*80)
        
        print(f"\nPipeline Results: {self.results_file}")
        print(f"Ground Truth:     {self.groundtruth_file}")
        
        print("\n" + "-"*80)
        print("OVERALL METRICS")
        print("-"*80)
        print(f"Precision:        {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:           {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1 Score:         {metrics['f1']:.4f}")
        
        print("\n" + "-"*80)
        print("DETAILED COUNTS")
        print("-"*80)
        print(f"True Positives:   {metrics['true_positives']}")
        print(f"False Positives:  {metrics['false_positives']}")
        print(f"False Negatives:  {metrics['false_negatives']}")
        
        print("\n" + "-"*80)
        print("PAPER COVERAGE")
        print("-"*80)
        print(f"Papers in Ground Truth:    {metrics['groundtruth_papers']}")
        print(f"Papers in Pipeline Results: {metrics['predicted_papers']}")
        print(f"Papers in Both:            {metrics['common_papers']}")
        
        # Show per-paper breakdown for common papers
        detailed = self.get_detailed_results()
        if detailed:
            print("\n" + "-"*80)
            print("PER-PAPER RESULTS (Common Papers)")
            print("-"*80)
            
            for pmc_id in sorted(detailed.keys())[:10]:  # Show first 10
                info = detailed[pmc_id]
                print(f"\n{pmc_id}:")
                print(f"  Precision: {info['precision']:.2f}, Recall: {info['recall']:.2f}")
                print(f"  TP: {len(info['true_positives'])}, FP: {len(info['false_positives'])}, FN: {len(info['false_negatives'])}")
                
                if info['true_positives']:
                    print(f"  ✓ Correct: {', '.join(info['true_positives'][:3])}" + 
                          (f" (+{len(info['true_positives'])-3} more)" if len(info['true_positives']) > 3 else ""))
                if info['false_positives']:
                    print(f"  ✗ Extra:   {', '.join(info['false_positives'][:3])}" +
                          (f" (+{len(info['false_positives'])-3} more)" if len(info['false_positives']) > 3 else ""))
                if info['false_negatives']:
                    print(f"  ✗ Missed:  {', '.join(info['false_negatives'][:3])}" +
                          (f" (+{len(info['false_negatives'])-3} more)" if len(info['false_negatives']) > 3 else ""))
            
            if len(detailed) > 10:
                print(f"\n... and {len(detailed) - 10} more papers")
        
        print("\n" + "="*80)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("DocETL Pipeline Evaluation")
        print("\nUsage:")
        print("  python scripts/eval.py --xml       Evaluate XML pipeline (EXP ground truth)")
        print("  python scripts/eval.py --pdf       Evaluate PDF pipeline")
        print("  python scripts/eval.py <results.json> <groundtruth.csv>")
        print("\nExamples:")
        print("  python scripts/eval.py --xml")
        print("  python scripts/eval.py data/output/xml_results/xml_dataset_references_output.json data/groundtruth/EXP_groundtruth.csv")
        sys.exit(1)
    
    # Predefined shortcuts
    if sys.argv[1] == '--xml':
        results_file = 'data/output/xml_results/xml_dataset_references_output.json'
        groundtruth_file = 'data/groundtruth/EXP_groundtruth.csv'
    elif sys.argv[1] == '--pdf':
        results_file = 'data/output/pdf_results/dataset_references_output.json'
        groundtruth_file = 'data/groundtruth/EXP_groundtruth.csv'
    else:
        if len(sys.argv) < 3:
            print("Error: Please provide both results file and ground truth file")
            sys.exit(1)
        results_file = sys.argv[1]
        groundtruth_file = sys.argv[2]
    
    # Check files exist
    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    if not Path(groundtruth_file).exists():
        print(f"Error: Ground truth file not found: {groundtruth_file}")
        sys.exit(1)
    
    # Run evaluation
    evaluator = PipelineEvaluator(results_file, groundtruth_file)
    evaluator.print_evaluation_report()


if __name__ == "__main__":
    main()
