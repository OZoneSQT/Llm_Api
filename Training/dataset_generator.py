from datasets import Dataset
import PyPDF2
import ebooklib
from ebooklib import epub
import mobi
import glob
import os
import docx
import textract
import datetime


######################
### Setup datasets ###
######################

# Local data

script_dir = os.path.dirname(os.path.abspath(__file__))
docs_folder = os.path.join(script_dir, "docs_folder")
output_dir = os.path.join(script_dir, "data")
os.makedirs(output_dir, exist_ok=True)

samples = []

# Load TXT files (each file as one document)
for txt_path in glob.glob(os.path.join(docs_folder, "*.txt")) + glob.glob(os.path.join(docs_folder, "*.TXT")):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if text:
            samples.append(text)

# Load PDF files (each file as one document)
for pdf_path in glob.glob(os.path.join(docs_folder, "*.pdf")) + glob.glob(os.path.join(docs_folder, "*.PDF")):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        if text.strip():
            samples.append(text.strip())

# Load EPUB files (each file as one document)
for epub_path in glob.glob(os.path.join(docs_folder, "*.epub")) + glob.glob(os.path.join(docs_folder, "*.EPUB")):
    book = epub.read_epub(epub_path)
    text = ""
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            text += item.get_body_content_str()
    if text.strip():
        samples.append(text.strip())

# Load MOBI files (each file as one document)
for mobi_path in glob.glob(os.path.join(docs_folder, "*.mobi")) + glob.glob(os.path.join(docs_folder, "*.MOBI")):
    book = mobi.Mobi(mobi_path)
    book.parse()

# Load DOC (each file as one document)
for doc_path in glob.glob(os.path.join(docs_folder, "*.doc")) + glob.glob(os.path.join(docs_folder, "*.DOC")):
    text = ""
    # Check if textract is available
    try:
        textract_available = True
        _ = textract.process  # Just to check if textract is imported
    except Exception:
        textract_available = False

    if textract_available:
        try:
            text = textract.process(doc_path).decode("utf-8").strip()
        except Exception as e:
            print(f"Failed to read {doc_path} with textract: {e}")
    else:
        print(f"Skipping {doc_path}: textract not available.")
    if text:
        samples.append(text)

# Load DOCX files (each file as one document)
for docx_path in glob.glob(os.path.join(docs_folder, "*.docx")) + glob.glob(os.path.join(docs_folder, "*.DOCX")):
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs]).strip()
        if text:
            samples.append(text)
    except Exception as e:
        print(f"Failed to read {docx_path}: {e}")

# Warning if no documents found
if not samples:
    print("No documents found.")

# Remove duplicates while preserving order, then create Dataset with a "text" column.
if samples:
    unique_samples = list(dict.fromkeys(samples))
    custom_dataset = Dataset.from_dict({"text": unique_samples})
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    custom_dataset.save_to_disk(f"./datasets/custom_dataset_{timestamp}")
