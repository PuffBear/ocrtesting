# use this file to extract content from tables.

import cv2
import pytesseract
import json
import numpy as np
import os

def detect_table_grid(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_bin = cv2.adaptiveThreshold(~img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 15, -2)

    # Detect horizontal lines
    horizontal = img_bin.copy()
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1] // 20, 1))
    horizontal = cv2.erode(horizontal, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)

    # Detect vertical lines
    vertical = img_bin.copy()
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[0] // 20))
    vertical = cv2.erode(vertical, vert_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vert_kernel, iterations=1)

    # Combine for table mask
    table_mask = cv2.add(horizontal, vertical)

    # Contour detection
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 40 and h > 20:  # Skip small boxes
            boxes.append((x, y, w, h))

    # Sort top-to-bottom, then left-to-right within rows
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes, img_bin

def extract_text_from_boxes(image_path, boxes):
    img_color = cv2.imread(image_path)
    results = []
    current_row = []
    last_y = -1

    for (x, y, w, h) in boxes:
        if last_y != -1 and abs(y - last_y) > 10:
            results.append(current_row)
            current_row = []

        roi = img_color[y:y+h, x:x+w]
        config = r'-l eng+hin --psm 6'
        text = pytesseract.image_to_string(roi, config=config).strip()
        current_row.append(text if text else "")
        last_y = y

    if current_row:
        results.append(current_row)

    return results

def process_table_image(image_path, output_json="structured_table_output.json"):
    boxes, _ = detect_table_grid(image_path)
    table_data = extract_text_from_boxes(image_path, boxes)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(table_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Saved to {output_json}")
    return table_data

# Example run
if __name__ == "__main__":
    image_path = "/Users/Agriya/Desktop/ocrtesting/data/intermediate/fir2017/page_004.png"
    output_json_path = "output/structured_page_004.json"
    structured = process_table_image(image_path, output_json_path)
