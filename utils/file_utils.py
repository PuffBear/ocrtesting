import os
from pdf2image import convert_from_path
from typing import List


def convert_pdf_to_images(filename: str, dpi: int = 300) -> List[str]:
    """
    Converts a PDF (from input/) to a list of image file paths.

    Args:
        filename (str): PDF filename (e.g., 'fir2017.pdf').
        dpi (int): Resolution for image conversion.

    Returns:
        List[str]: List of saved image file paths (in reading order).
    """
    input_path = os.path.join("input", filename)
    assert os.path.exists(input_path), f"File does not exist: {input_path}"

    pdf_name = os.path.splitext(filename)[0]
    output_dir = os.path.join("data", "intermediate", pdf_name)
    os.makedirs(output_dir, exist_ok=True)

    images = convert_from_path(input_path, dpi=dpi, poppler_path='/opt/homebrew/bin')

    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i + 1:03d}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)

    return image_paths
