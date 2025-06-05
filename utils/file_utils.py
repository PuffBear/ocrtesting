def convert_pdf_to_images(input_path, dpi=300, output_folder="data/intermediate/fir2017/"):
    import os
    from pdf2image import convert_from_path

    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(input_path, dpi=dpi)
    image_paths = []

    for idx, img in enumerate(images):
        img_path = os.path.join(output_folder, f"page_{idx + 1:03d}.png")
        img.save(img_path, "PNG")
        image_paths.append(img_path)

    return image_paths
