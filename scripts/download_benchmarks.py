"""Download DataRef-EXP / DataRef-REV from Zenodo.

The benchmark is published at https://doi.org/10.5281/zenodo.15549086.
Zenodo serves a JSON record listing files; we fetch each file into
data/benchmark/. If the network or the record layout changes, the script
fails gracefully and prints manual download instructions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ZENODO_RECORD_ID = "15549086"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
TARGET_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmark"

INSTRUCTIONS = f"""
Manual download instructions
============================
1. Open https://doi.org/10.5281/zenodo.15549086 in a browser.
2. Download every file from that record.
3. Place them under {TARGET_DIR}.
4. Re-run scripts/evaluate_docetl.py once the ground-truth CSV is in place.
"""


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(ZENODO_API, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Could not contact Zenodo API: {e}")
        print(INSTRUCTIONS)
        return 1
    record = resp.json()
    files = record.get("files", [])
    if not files:
        print("[ERROR] Zenodo record returned no files.")
        print(INSTRUCTIONS)
        return 1

    manifest = []
    for f in files:
        name = f.get("key") or f.get("filename") or "unknown"
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        if not url:
            print(f"[WARN] Skipping {name}: no download URL")
            continue
        target = TARGET_DIR / name
        print(f"Downloading {name} → {target}")
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with target.open("wb") as out:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        out.write(chunk)
            manifest.append({"name": name, "size": target.stat().st_size})
        except Exception as e:
            print(f"[WARN] Failed to download {name}: {e}")

    (TARGET_DIR / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {len(manifest)} files saved to {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
