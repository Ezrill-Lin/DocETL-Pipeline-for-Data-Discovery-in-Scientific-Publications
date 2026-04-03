# DocETL Parsing Packages Explained

## What is `docetl[parsing]`?

`docetl[parsing]` is an "extra" dependency set that installs **built-in parsing tools** for common document formats.

## What's INCLUDED in `docetl[parsing]`:

✅ **Excel files** (.xlsx) - via `openpyxl`
✅ **Word documents** (.docx) - via `python-docx`  
✅ **PowerPoint** (.pptx) - via `python-pptx`
✅ **Text files** (.txt, .md) - built-in
✅ **Basic PDF text extraction** - via `PyMuPDF` (for PDFs with selectable text)

These are the "lightweight" parsers that work well for most use cases.

## What's NOT INCLUDED (Optional):

❌ **PaddleOCR** - For scanned PDFs/images (not installed by default because it's ~500MB)
❌ **Azure Document Intelligence** - Cloud-based PDF parsing (requires Azure account)
❌ **Whisper** - Audio transcription (requires separate installation)

## For Your Scientific Papers Pipeline:

### Do you need to install a PDF parser?

**It depends on your PDFs:**

1. **If your PDFs have selectable text** (you can copy/paste text from them):
   - ✅ You're good! PyMuPDF (already included) can extract the text
   - No additional installation needed
   
2. **If your PDFs are scanned images** (OCR needed):
   - ❌ You need to install PaddleOCR:
   ```bash
   pip install paddlepaddle paddleocr
   ```
   - OR use Azure Document Intelligence (faster, more accurate, but requires Azure account)

## Quick Test:

To check if your PDFs work with the current setup:

```bash
python etl.py run
```

If you get an error about "paddleocr not found", then your PDFs need OCR and you should install it.

If it works but extracts blank/no text, your PDFs are scanned images and you need PaddleOCR.

## Recommended Approach:

1. **Try running the pipeline first** with what you have
2. **If text extraction fails**, install PaddleOCR
3. **For production**, consider Azure DI (much faster than PaddleOCR)

## Alternative: Custom PDF Parser

You can also create a simpler custom parser using PyMuPDF (already installed):

```yaml
parsing_tools:
  - name: simple_pdf_parser
    function_code: |
      def simple_pdf_parser(document: Dict) -> List[Dict]:
          import fitz  # PyMuPDF
          
          pdf_path = document["pdf_path"]
          doc = fitz.open(pdf_path)
          
          text = ""
          for page in doc:
              text += page.get_text()
          
          doc.close()
          
          return [{"paper_content": text}]

datasets:
  scientific_papers:
    type: file
    source: local
    path: "papers_input.json"
    parsing:
      - function: simple_pdf_parser
```

This uses PyMuPDF which is **much faster** than PaddleOCR but only works for PDFs with selectable text.
