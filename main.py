from utils.file_utils import convert_pdf_to_images
print("Starting.")
input_pdf_path = "/Users/Agriya/Desktop/ocrtesting/input/fir2016.pdf"
print("converting.")
image_paths = convert_pdf_to_images(input_pdf_path)
print("✅ Generated images:")
for path in image_paths:
    print(f" - {path}")