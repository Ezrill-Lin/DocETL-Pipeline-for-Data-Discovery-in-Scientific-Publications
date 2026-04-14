"""
DocETL Pipeline Runner - Main Entry Point

This script runs the DocETL pipelines for extracting dataset references 
from scientific papers (PDF, HTML, or XML formats).

Usage:
    python main.py pdf              # Run PDF/HTML pipeline
    python main.py xml              # Run XML pipeline
    python main.py pdf --optimize   # Run with optimization
    python main.py all              # Run both pipelines
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional


class PipelineRunner:
    """Main runner for DocETL pipelines."""
    
    def __init__(self):
        self.pipelines_dir = Path(__file__).parent
        self.pdf_pipeline = self.pipelines_dir / "pipeline_pdf.yaml"
        self.xml_pipeline = self.pipelines_dir / "pipeline_xml.yaml"
    
    def run_pipeline(self, pipeline_path: Path, optimize: bool = False) -> bool:
        """
        Run a DocETL pipeline.
        
        Args:
            pipeline_path: Path to the pipeline YAML file
            optimize: Whether to run with optimization
            
        Returns:
            True if successful, False otherwise
        """
        if not pipeline_path.exists():
            print(f"❌ Pipeline not found: {pipeline_path}")
            return False
        
        print(f"\n{'='*70}")
        print(f"🚀 Running pipeline: {pipeline_path.name}")
        print(f"{'='*70}\n")
        
        # Build command
        cmd = ["docetl", "run", str(pipeline_path)]
        if optimize:
            cmd.insert(2, "--optimize")
        
        # Run the pipeline
        try:
            result = subprocess.run(
                cmd,
                cwd=self.pipelines_dir,
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            if result.returncode == 0:
                print(f"\n{'='*70}")
                print(f"✅ Pipeline completed successfully!")
                print(f"{'='*70}\n")
                return True
            else:
                print(f"\n{'='*70}")
                print(f"❌ Pipeline failed with exit code {result.returncode}")
                print(f"{'='*70}\n")
                return False
                
        except FileNotFoundError:
            print("\n❌ Error: 'docetl' command not found.")
            print("Make sure you've activated the virtual environment:")
            print("  & '.\\docetl_venv\\Scripts\\Activate.ps1'  (Windows)")
            print("  source docetl_venv/bin/activate  (Linux/Mac)")
            return False
        except Exception as e:
            print(f"\n❌ Error running pipeline: {e}")
            return False
    
    def run_pdf_pipeline(self, optimize: bool = False) -> bool:
        """Run the PDF/HTML pipeline."""
        print("\n📄 PDF/HTML Pipeline")
        print("Input: data/input/pdf/")
        print("Output: data/output/pdf_results/dataset_references_output.json")
        return self.run_pipeline(self.pdf_pipeline, optimize)
    
    def run_xml_pipeline(self, optimize: bool = False) -> bool:
        """Run the XML pipeline."""
        print("\n📋 XML Pipeline")
        print("Input: data/input/xml/")
        print("Output: data/output/xml_results/xml_dataset_references_output.json")
        return self.run_pipeline(self.xml_pipeline, optimize)
    
    def run_all(self, optimize: bool = False) -> None:
        """Run all pipelines sequentially."""
        print("\n🔄 Running all pipelines...\n")
        
        success_count = 0
        total_count = 2
        
        if self.run_pdf_pipeline(optimize):
            success_count += 1
        
        if self.run_xml_pipeline(optimize):
            success_count += 1
        
        print(f"\n{'='*70}")
        print(f"📊 Summary: {success_count}/{total_count} pipelines completed successfully")
        print(f"{'='*70}\n")


def print_usage():
    """Print usage information."""
    print("\n" + "="*70)
    print("DocETL Pipeline Runner")
    print("="*70)
    print("\nUsage:")
    print("  python main.py pdf              Run PDF/HTML pipeline")
    print("  python main.py xml              Run XML pipeline")
    print("  python main.py all              Run both pipelines")
    print("  python main.py pdf --optimize   Run with optimization")
    print("\nPipelines:")
    print("  • PDF/HTML: Extracts datasets from PDF and HTML papers")
    print("  • XML:      Extracts datasets from PubMed Central XML papers")
    print("\nUtility Scripts (in ../scripts/):")
    print("  • utils.py:          Analyze and export pipeline results")
    print("  • download_papers.py: Download papers from various sources")
    print("\nExamples:")
    print("  python main.py pdf")
    print("  python main.py xml --optimize")
    print("  python main.py all")
    print("\nNote: Make sure the docetl virtual environment is activated!")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    runner = PipelineRunner()
    command = sys.argv[1].lower()
    
    # Check for --optimize flag
    optimize = "--optimize" in sys.argv or "-o" in sys.argv
    
    if command == "pdf":
        success = runner.run_pdf_pipeline(optimize)
        sys.exit(0 if success else 1)
    
    elif command == "xml":
        success = runner.run_xml_pipeline(optimize)
        sys.exit(0 if success else 1)
    
    elif command == "all":
        runner.run_all(optimize)
        sys.exit(0)
    
    else:
        print(f"❌ Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
