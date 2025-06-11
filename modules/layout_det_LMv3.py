from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch
import pytesseract
import os

# Setup tesseract path and tessdata
pytesseract.pytesseract.tesseract_cmd = "/home/austere/Testing/ocrtesting/venv/src/tesseract/bin/tesseract"
os.environ["TESSDATA_PREFIX"] = "/home/austere/Testing/ocrtesting/venv/src/tesseract/tessdata"

# Load image
image_path = "/home/austere/Testing/ocrtesting/data/intermediate/fir2017/page_002.png"
image = Image.open(image_path).convert("RGB")

# Load model and processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# Get encoding from image processor (with OCR lang)
encoding = processor.image_processor(images=image, return_tensors="pt", ocr_lang="eng+hin")
# Then tokenize the OCR output
encoding = processor.tokenizer(
    encoding["words"],
    boxes=encoding["boxes"],
    return_tensors="pt",
    truncation=True,
    padding="max_length"
)

# Forward pass
with torch.no_grad():
    outputs = model(**encoding)
    predictions = torch.argmax(outputs.logits, dim=2)

# Decode tokens
tokens = processor.tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
labels = predictions[0].numpy()

# Print result
for token, label in zip(tokens, labels):
    print(f"{token}: label {label}")
