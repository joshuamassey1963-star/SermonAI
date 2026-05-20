from pypdf import PdfReader
import pdfplumber
from pathlib import Path
import re

PDF_PATH = Path.home() / "SermonAI/data/pdfs/test.pdf"
OUTPUT_PATH = Path.home() / "SermonAI/data/output/test.txt"

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\x00', '')
    return text.strip()

def extract_with_pypdf(pdf_path):
    text = ""

    reader = PdfReader(str(pdf_path))

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

def extract_with_pdfplumber(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text

def main():
    print("Starting extraction...")

    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    try:
        text = extract_with_pypdf(PDF_PATH)

        if len(text.strip()) < 100:
            print("Low text detected, switching to pdfplumber...")
            text = extract_with_pdfplumber(PDF_PATH)

        cleaned = clean_text(text)

        OUTPUT_PATH.write_text(cleaned, encoding="utf-8")

        print(f"Extraction complete.")
        print(f"Saved to: {OUTPUT_PATH}")
        print(f"Characters extracted: {len(cleaned)}")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
