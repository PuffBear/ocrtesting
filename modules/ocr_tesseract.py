# ocr_simple.py
# use this script to extract multilingual text.

import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import re

# Optional: Set path if not auto-detected
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# Add Hindi language support
custom_config = r'-l eng+hin --oem 1 --psm 6'

# Load image
img_path = "/Users/Agriya/Desktop/ocrtesting/data/intermediate/fir2020/page_001.png"
img = cv2.imread(img_path)

# Convert to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# OCR
text = pytesseract.image_to_string(rgb, config=custom_config)
print("🔍 Extracted Text:\n", text)

# Extract page number (e.g., page_002 → 002)
match = re.search(r'page_(\d+)', img_path)
page_number = match.group(1) if match else "unknown"

# Ensure output directory exists
output_dir = "output/2020"
os.makedirs(output_dir, exist_ok=True)

# Save to file
output_path = os.path.join(output_dir, f"{page_number}_output.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"✅ Text saved to {output_path}")