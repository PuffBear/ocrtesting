# ocr_simple.py

import pytesseract
from PIL import Image
import cv2
import numpy as np

# Optional: Set path if not auto-detected
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# Add Hindi language support
custom_config = r'-l eng+hin --oem 1 --psm 6'

# Load image
img_path = "data/intermediate/fir2017/page_002.png"
img = cv2.imread(img_path)

# Convert to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# OCR
text = pytesseract.image_to_string(rgb, config=custom_config)
print("🔍 Extracted Text:\n", text)
