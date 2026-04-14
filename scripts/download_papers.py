"""
Download scientific papers from PMC using NCBI E-utilities API.
Papers are saved as XML format with full structured content.
"""
import os
import csv
import requests
import time
from pathlib import Path
from typing import Tuple, Optional, Set

def extract_pmc_id(url: str) -> Optional[str]:
    """Extract PMC ID number from URL."""
    if "PMC" in url:
        parts = url.split("PMC")
        if len(parts) > 1:
            pmc_num = parts[1].strip("/").split("/")[0]
            return pmc_num
    return None

def download_via_eutils(pmc_id: str, output_folder: str) -> Tuple[bool, str, str]:
    """Download paper using NCBI E-utilities API.
    
    Args:
        pmc_id: PMC ID number (without 'PMC' prefix)
        output_folder: Folder to save the downloaded paper
        
    Returns:
        Tuple of (success, format_type, result_message)
    """
    try:
        # Use E-utilities efetch to get the full text XML
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'pmc',
            'id': pmc_id,
            'rettype': 'xml',
            'retmode': 'xml'
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Save as XML (contains full structured article)
        output_path = os.path.join(output_folder, f"PMC{pmc_id}.xml")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        return (True, 'xml', output_path)
    except Exception as e:
        return (False, 'error', str(e)[:80])

def main() -> None:
    """Main function to download PMC papers from CSV file."""
    # Configuration
    csv_file = "EXP_groundtruth.csv"
    output_folder = "exp_papers"
    
    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    print(f"Created/verified output folder: {output_folder}\n")
    
    # Read CSV and extract unique PMC IDs
    unique_pmc_ids: Set[str] = set()
    try:
        with open(csv_file, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get('citing_publication_link', '').strip()
                if url and url != 'N/A' and 'pmc' in url.lower():
                    pmc_num = extract_pmc_id(url)
                    if pmc_num:
                        unique_pmc_ids.add(pmc_num)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    print(f"Found {len(unique_pmc_ids)} unique PMC article(s) to download\n")
    
    # Download papers
    success_count = 0
    failed_count = 0
    
    for idx, pmc_id in enumerate(sorted(unique_pmc_ids), 1):
        filename = f"PMC{pmc_id}.xml"
        output_path = os.path.join(output_folder, filename)
        
        # Skip if already downloaded
        if os.path.exists(output_path):
            print(f"[{idx}/{len(unique_pmc_ids)}] Already exists: {filename}")
            success_count += 1
            continue
        
        print(f"[{idx}/{len(unique_pmc_ids)}] Downloading PMC{pmc_id}...")
        
        success, format_type, result = download_via_eutils(pmc_id, output_folder)
        
        if success:
            print(f"  ✅ Saved as XML: {filename}")
            success_count += 1
        else:
            print(f"  ❌ Failed: {result}")
            failed_count += 1
        
        # Be nice to NCBI's servers - they request max 3 requests/second
        if idx < len(unique_pmc_ids):
            time.sleep(0.4)
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary:")
    print(f"  ✅ Successful: {success_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  📁 Output folder: {output_folder}")
    print("\nNote: Papers saved as XML format with full structured text")
    print("="*60)

if __name__ == "__main__":
    main()
