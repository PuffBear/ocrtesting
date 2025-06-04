# fir_ocr_pipeline/main.py
import os
import io
import json
from google.cloud import vision
from pdf2image import convert_from_path

# === SETUP ===
# Replace this with your actual key path or set GOOGLE_APPLICATION_CREDENTIALS env
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your-service-account.json"

client = vision.ImageAnnotatorClient()


def extract_text_blocks_from_image(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    img = vision.Image(content=buf.getvalue())
    response = client.document_text_detection(image=img)
    annotation = response.full_text_annotation

    text_blocks = []
    for page in annotation.pages:
        for block in page.blocks:
            block_text = ""
            for para in block.paragraphs:
                for word in para.words:
                    block_text += ''.join([symbol.text for symbol in word.symbols]) + ' '
            vertices = [
                {"x": v.x, "y": v.y} for v in block.bounding_box.vertices
            ]
            text_blocks.append({
                "text": block_text.strip(),
                "bounding_box": vertices
            })

    return annotation.text, text_blocks


def parse_key_fields(text):
    import re
    fields = {
        "fir_number": None,
        "fir_date": None,
        "ipc_sections": [],
    }
    if m := re.search(r"(?:क्रा?मांक|FIR\s*No\.?)[\s:]*([\w\-\/]+)", text):
        fields["fir_number"] = m.group(1)
    if m := re.search(r"(?:दिनांक|Date)[\s:]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})", text):
        fields["fir_date"] = m.group(1)
    if m := re.findall(r"धारा[\s:]*([\d, ]+)", text):
        fields["ipc_sections"] = [s.strip() for s in m[0].split(',')]
    return fields


def process_pdf(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    doc_data = {
        "document": os.path.basename(pdf_path),
        "pages": [],
        "full_text": ""
    }

    for idx, page in enumerate(pages):
        text, blocks = extract_text_blocks_from_image(page)
        key_fields = parse_key_fields(text)
        doc_data["pages"].append({
            "page_number": idx + 1,
            "text_blocks": blocks,
            "key_fields": key_fields
        })
        doc_data["full_text"] += text + "\n\n--- PAGE BREAK ---\n\n"

    return doc_data


if __name__ == "__main__":
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            print(f"Processing {filename}...")
            data = process_pdf(os.path.join(INPUT_DIR, filename))
            with open(os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".json")), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved structured JSON for {filename}\n")
