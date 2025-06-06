from utils.convert_file import convert_pdf_to_images

input_pdf_path = "/Users/Agriya/Desktop/ocrtesting/input/fir2020casefile.pdf"
image_paths = convert_pdf_to_images(input_pdf_path)

print("✅ Generated images:")
for path in image_paths:
    print(f" - {path}")