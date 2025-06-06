from paddleocr import PaddleOCR
from paddleocr import draw_ocr  # Try again now
from PIL import Image
import numpy as np
import cv2
import os

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# Input path
img_path = 'data/intermediate/fir2017/page_001.png'
image = Image.open(img_path).convert('RGB')

# Run OCR
result = ocr.ocr(img_path, cls=True)

# Extract results
boxes = [line[0] for line in result[0]]
txts = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]

# Draw results
img_array = np.array(image)
annotated = draw_ocr(img_array, boxes, txts, scores)

# Save
os.makedirs("output", exist_ok=True)
cv2.imwrite("output/ocr_result.jpg", annotated)
