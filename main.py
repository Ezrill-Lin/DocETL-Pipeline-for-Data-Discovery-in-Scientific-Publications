"""
DocETL pipeline runner.

This runner targets the project's local ``docetl_venv`` executable, refreshes
the PDF/XML manifests, and launches the selected pipeline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


class PipelineRunner:
    """Main runner for DocETL pipelines."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent
        self.pipelines_dir = self.project_root / "pipelines"
        self.data_input_dir = self.project_root / "data" / "input"
        self.env_file = self.project_root / ".env"
        self.pdf_pipeline = self.pipelines_dir / "pipeline_pdf.yaml"
        self.xml_pipeline = self.pipelines_dir / "pipeline_xml.yaml"
        self.docetl_exe = self.project_root / "docetl_venv" / "Scripts" / "docetl.exe"
        self.pdf_manifest = self.data_input_dir / "pdf" / "papers_input.json"
        self.xml_manifest = self.data_input_dir / "xml" / "xml_papers_input.json"
        self.max_threads = self._read_max_threads()

    def load_dotenv(self) -> None:
        """Load simple KEY=VALUE pairs from ``.env`` without extra dependencies."""
        if not self.env_file.exists():
            return

        for raw_line in self.env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value

    def _read_max_threads(self) -> int | None:
        value = os.getenv("DOCETL_MAX_THREADS", "").strip()
        if not value:
            return None

        try:
            parsed = int(value)
        except ValueError:
            print(f"Warning: invalid DOCETL_MAX_THREADS value: {value!r}. Ignoring.")
            return None

        if parsed < 1:
            print(f"Warning: DOCETL_MAX_THREADS must be >= 1. Ignoring {parsed}.")
            return None

        return parsed

    def ensure_environment(self) -> bool:
        """Validate that the local DocETL runtime can be executed."""
        self.load_dotenv()

        if not self.docetl_exe.exists():
            print("\nError: project DocETL executable not found.")
            print(f"Expected: {self.docetl_exe}")
            print("Create or restore the virtual environment in `docetl_venv`.")
            return False

        if not os.getenv("OPENAI_API_KEY"):
            print("\nError: OPENAI_API_KEY is not set.")
            print("Set it in your shell or create a project `.env` file.")
            print("See `.env.example` for the expected variable name.")
            return False

        return True

    def refresh_manifests(self) -> None:
        """Regenerate the JSON manifests expected by the pipelines."""
        pdf_count = self._refresh_pdf_manifest()
        xml_count = self._refresh_xml_manifest()
        print(f"Refreshed manifests: {pdf_count} PDF files, {xml_count} XML files.")

    def _refresh_pdf_manifest(self) -> int:
        pdf_dir = self.data_input_dir / "pdf"
        records = []
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            records.append(
                {
                    "paper_id": pdf_path.stem,
                    "title": pdf_path.stem,
                    "pdf_path": f"../data/input/pdf/{pdf_path.name}",
                }
            )

        with self.pdf_manifest.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
            handle.write("\n")

        return len(records)

    def _refresh_xml_manifest(self) -> int:
        xml_dir = self.data_input_dir / "xml"
        records = []
        for xml_path in sorted(xml_dir.glob("*.xml")):
            records.append(
                {
                    "paper_id": xml_path.stem,
                    "title": xml_path.stem,
                    "xml_path": f"../data/input/xml/{xml_path.name}",
                }
            )

        with self.xml_manifest.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
            handle.write("\n")

        return len(records)

    def run_pipeline(self, pipeline_path: Path) -> bool:
        """Run a DocETL pipeline using the project-local executable."""
        if not pipeline_path.exists():
            print(f"Pipeline not found: {pipeline_path}")
            return False

        print(f"\n{'=' * 70}")
        print(f"Running pipeline: {pipeline_path.name}")
        print(f"{'=' * 70}\n")

        cmd = [str(self.docetl_exe), "run", str(pipeline_path.name)]
        if self.max_threads is not None:
            cmd.extend(["--max-threads", str(self.max_threads)])

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        try:
            result = subprocess.run(
                cmd,
                cwd=self.pipelines_dir,
                capture_output=False,
                text=True,
                check=False,
                env=env,
            )
        except Exception as error:
            print(f"\nError running pipeline: {error}")
            return False

        if result.returncode == 0:
            print(f"\n{'=' * 70}")
            print("Pipeline completed successfully.")
            print(f"{'=' * 70}\n")
            return True

        print(f"\n{'=' * 70}")
        print(f"Pipeline failed with exit code {result.returncode}")
        print(f"{'=' * 70}\n")
        return False

    def run_pdf_pipeline(self) -> bool:
        """Run the PDF pipeline."""
        print("\nPDF Pipeline")
        print("Input manifest: data/input/pdf/papers_input.json")
        print("Output: data/output/pdf_results/dataset_references_output.json")
        return self.run_pipeline(self.pdf_pipeline)

    def run_xml_pipeline(self) -> bool:
        """Run the XML pipeline."""
        print("\nXML Pipeline")
        print("Input manifest: data/input/xml/xml_papers_input.json")
        print("Output: data/output/xml_results/xml_dataset_references_output.json")
        return self.run_pipeline(self.xml_pipeline)

    def run_all(self) -> bool:
        """Run both pipelines sequentially and report success."""
        print("\nRunning all pipelines...\n")

        success_count = 0
        total_count = 2

        if self.run_pdf_pipeline():
            success_count += 1

        if self.run_xml_pipeline():
            success_count += 1

        print(f"\n{'=' * 70}")
        print(f"Summary: {success_count}/{total_count} pipelines completed successfully")
        print(f"{'=' * 70}\n")
        return success_count == total_count


def print_usage() -> None:
    """Print usage information."""
    print("\n" + "=" * 70)
    print("DocETL Pipeline Runner")
    print("=" * 70)
    print("\nUsage:")
    print("  python main.py pdf   Run PDF pipeline")
    print("  python main.py xml   Run XML pipeline")
    print("  python main.py all   Run both pipelines")
    print("  python main.py help  Show this message")
    print("\nNotes:")
    print("  - Uses the project-local `docetl_venv` executable")
    print("  - Regenerates PDF/XML input manifests before each run")
    print("  - Loads `.env` if present and OPENAI_API_KEY is not already set")
    print("  - Accepts optional DOCETL_MAX_THREADS in the environment")
    print("=" * 70 + "\n")


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()
    if command in {"help", "--help", "-h"}:
        print_usage()
        sys.exit(0)

    runner = PipelineRunner()

    if not runner.ensure_environment():
        sys.exit(1)

    runner.refresh_manifests()

    if command == "pdf":
        sys.exit(0 if runner.run_pdf_pipeline() else 1)

    if command == "xml":
        sys.exit(0 if runner.run_xml_pipeline() else 1)

    if command == "all":
        sys.exit(0 if runner.run_all() else 1)

    print(f"Unknown command: {command}")
    print_usage()
    sys.exit(1)


if __name__ == "__main__":
    main()
