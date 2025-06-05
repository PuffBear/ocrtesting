# main.py

import os
from utils.file_utils import convert_pdf_to_images
from modules.ocr_paddle import init_paddle_ocr, run_ocr_on_image
from utils.json_utils import save_json

PDF_FILE = "input/fir2017.pdf"  # <== your test file path
INTERMEDIATE_DIR = "data/intermediate/fir2017/"
OUTPUT_DIR = "output/fir2017_ocr_output.json"

if __name__ == "__main__":
    # Step 1: Convert PDF to images
    image_list = convert_pdf_to_images(PDF_FILE, output_folder=INTERMEDIATE_DIR)
    print("✅ Converted PDF to images:", image_list)

    # Step 2: Initialize OCR model
    ocr_model = init_paddle_ocr()

    # Step 3: Run OCR on each image
    final_results = {}
    for img_path in image_list:
        print(f"🔍 OCR on: {img_path}")
        ocr_blocks = run_ocr_on_image(ocr_model, img_path)
        final_results[img_path] = ocr_blocks

    # Step 4: Save results to JSON
    save_json(final_results, OUTPUT_DIR)
    print(f"\n📁 OCR results saved to: {OUTPUT_DIR}")
