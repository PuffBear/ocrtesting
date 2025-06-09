import cv2
import pytesseract
import os
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Paths
img_path = "/Users/Agriya/Desktop/ocrtesting/data/intermediate/fir2017/page_002.png"
output_path = "output/page_002_structured.json"
os.makedirs("output", exist_ok=True)

# Load image and model
doc = DocumentFile.from_images(img_path)
model = ocr_predictor(pretrained=True)
result = model(doc)

# Extract blocks and layout
blocks = result.export()['pages'][0]['blocks']
img = cv2.imread(img_path)
H, W, _ = img.shape

# Extract line boxes
line_boxes = []
for block in blocks:
    for line in block.get('lines', []):
        word_boxes = line['words']
        if not word_boxes:
            continue
        x0 = min(word['geometry'][0][0] for word in word_boxes)
        y0 = min(word['geometry'][0][1] for word in word_boxes)
        x1 = max(word['geometry'][1][0] for word in word_boxes)
        y1 = max(word['geometry'][1][1] for word in word_boxes)
        line_boxes.append({
            "coords": [int(x0*W), int(y0*H), int(x1*W), int(y1*H)],
            "cy": (y0 + y1) / 2
        })

# Sort and OCR using Tesseract
line_boxes.sort(key=lambda b: b['cy'])
structured_rows = []
for box in line_boxes:
    x0, y0, x1, y1 = box['coords']
    cropped = img[y0:y1, x0:x1]
    text = pytesseract.image_to_string(cropped, config='--psm 6 -l eng+hin').strip()
    if text:
        structured_rows.append([text])
        

# Save output
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(structured_rows, f, indent=2, ensure_ascii=False)

print(f"✅ Structured output saved to: {output_path}")
