# using layoutLMv3 
# not using this. weird output. not worth it

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

# Load and preprocess the image
image_path = "/Users/Agriya/Desktop/ocrtesting/data/intermediate/fir2017/page_002.png"
image = Image.open(image_path).convert("RGB")

# Load processor and model
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# Preprocess: run OCR and get bounding boxes + tokens
encoding = processor(images=image, return_tensors="pt", truncation=True)

# Forward pass
with torch.no_grad():
    outputs = model(**encoding)
    predictions = torch.argmax(outputs.logits, dim=2)

# Decode results
tokens = processor.tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
labels = predictions[0].numpy()

# Print token-wise predictions (you’ll need label mapping if fine-tuned)
for token, label in zip(tokens, labels):
    print(f"{token}: label {label}")
