# need to finalise this script for layout detection for structured formatting.
# layout.py

import cv2
import pytesseract
import os
import json

pytesseract.pytesseract.tesseract_cmd = "/home/austere/Testing/ocrtesting/venv/src/tesseract/bin/tesseract"

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bin_img

def find_layout_regions(bin_img):
    # Dilation to merge lines of text into paragraphs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    dilated = cv2.dilate(bin_img, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 20:  # Filter out noise
            regions.append((x, y, w, h))
    return sorted(regions, key=lambda r: (r[1], r[0]))

def classify_region(img, x, y, w, h):
    roi = img[y:y+h, x:x+w]
    config = r'-l eng+hin --psm 6'
    text = pytesseract.image_to_string(roi, config=config).strip()

    # Heuristic for text density
    text_density = len(text) / (w * h) if w * h != 0 else 0

    # Heuristic for handwriting detection: low OCR accuracy, medium-sized region
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    num_edges = cv2.countNonZero(edges)
    edge_density = num_edges / (w * h) if w * h != 0 else 0

    # Heuristics
    if text_density < 0.0005 and edge_density > 0.05 and 50 < h < 300:
        region_type = "handwriting"
    elif w > 300 and h > 100:
        region_type = "table" if len(text.split()) < (w * h) // 1000 else "text_block"
    elif h < 50:
        region_type = "heading"
    else:
        region_type = "text_block"
    return region_type, text

def detect_layout(image_path, output_json="output/layout_output_2020_011.json"):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    img_color = cv2.imread(image_path)
    bin_img = preprocess_image(image_path)
    regions = find_layout_regions(bin_img)

    layout = []
    for (x, y, w, h) in regions:
        region_type, content = classify_region(img_color, x, y, w, h)
        layout.append({
            "type": region_type,
            "bbox": [x, y, x+w, y+h],
            "content": content
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)
    print(f"✅ Layout saved to {output_json}")

# Example run
if __name__ == "__main__":
    detect_layout("/home/austere/Testing/ocrtesting/data/intermediate/fir2020/page_011.png")
