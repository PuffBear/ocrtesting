# file_utils.py
import os
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

def convert_pdf_to_images(input_path, dpi=300, output_folder="data/intermediate/fir2021/"):
    import os
    from pdf2image import convert_from_path

    os.makedirs(output_folder, exist_ok=True)

    # Convert using Ghostscript (fallback for corrupt/complex scanned PDFs)
    images = convert_from_path(
        input_path,
        dpi=dpi,
        output_folder=None,  # don't write to disk yet
        fmt="png"
    )

    image_paths = []
    for idx, img in enumerate(images):
        img_path = os.path.join(output_folder, f"page_{idx + 1:03d}.png")
        img.save(img_path, "PNG")
        image_paths.append(img_path)

    return image_paths
