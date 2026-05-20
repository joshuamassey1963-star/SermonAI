from pathlib import Path
from pypdf import PdfReader
import pdfplumber
import re
import gc

PDF_DIR = Path.home() / "SermonAI/data/pdfs"
OUTPUT_DIR = Path.home() / "SermonAI/data/output"
FAILED_LOG = Path.home() / "SermonAI/data/failed_pdfs.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def repair_ocr_spacing(text):
    text = re.sub(r'(?<=\b\w) (?=\w\b)', '', text)
    return text

def clean_text(text):
    text = text.replace('\x00', '')

    text = repair_ocr_spacing(text)

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'\^+', '', text)

    text = text.strip()

    return text

def extract_pypdf(pdf_path):
    text = ""

    reader = PdfReader(str(pdf_path))

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

def extract_pdfplumber(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text

def process_pdf(pdf_path):
    print(f"\nProcessing: {pdf_path.name}")

    output_file = OUTPUT_DIR / f"{pdf_path.stem}.txt"

    try:
        text = extract_pypdf(pdf_path)

        if len(text.strip()) < 500:
            print("Fallback to pdfplumber...")
            text = extract_pdfplumber(pdf_path)

        cleaned = clean_text(text)

        output_file.write_text(cleaned, encoding="utf-8")

        print(f"Saved: {output_file.name}")
        print(f"Characters: {len(cleaned)}")

    except Exception as e:
        print(f"FAILED: {pdf_path.name}")
        print(f"ERROR: {e}")

        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(f"{pdf_path.name} | {e}\n")

    gc.collect()

def main():
    pdf_files = sorted(PDF_DIR.glob("*.[Pp][Dd][Ff]"))

    print(f"Found PDFs: {len(pdf_files)}")

    for index, pdf_file in enumerate(pdf_files, start=1):
        print(f"\n[{index}/{len(pdf_files)}]")
        process_pdf(pdf_file)

    print("\nDONE.")
    print(f"Output folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
