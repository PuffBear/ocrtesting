# ocr_to_single_file.py
import pytesseract
import cv2
import os
import re
import glob

# If needed, point pytesseract to your tesseract binary:
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# OCR config: English + Hindi
custom_config = r'-l eng+hin --oem 1 --psm 6'

# ─── INPUT & OUTPUT ────────────────────────────────────────────────────────────
input_dir  = "/Users/Agriya/Desktop/ocrtesting/data/intermediate/fir2016/"
output_dir = "/Users/Agriya/Desktop/ocrtesting/output/2016"
os.makedirs(output_dir, exist_ok=True)

# single output file
output_file = os.path.join(output_dir, "fir2016_full_output.txt")

# ─── PAGE-BREAK DELIMITER ─────────────────────────────────────────────────────
delimiter = (
    "-" * 150 + "\n"
    "xx" + " " * 70 + "Page Break" + " " * 70 + "xx\n"
    + "-" * 150 + "\n\n"
)

# ─── COLLECT ALL PAGES ─────────────────────────────────────────────────────────
# You can either glob or explicitly list pages 001–119:
all_pages = sorted(glob.glob(os.path.join(input_dir, "page_*.png")))
# Or, for exactly 001–119, uncomment:
# all_pages = [
#     os.path.join(input_dir, f"page_{i:03d}.png")
#     for i in range(1, 120)
# ]

# ─── RUN OCR & DUMP TO SINGLE FILE ──────────────────────────────────────────────
with open(output_file, "w", encoding="utf-8") as out_f:
    for img_path in all_pages:
        # load & convert
        img = cv2.imread(img_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # OCR
        text = pytesseract.image_to_string(rgb, config=custom_config)
        page_name = os.path.basename(img_path)

        # header for clarity (optional)
        out_f.write(f"Page: {page_name}\n\n")
        out_f.write(text.strip() + "\n\n")
        out_f.write(delimiter)

        print(f"Processed {page_name} → appended to single file")

print(f"\n✅ All text written to {output_file}")
