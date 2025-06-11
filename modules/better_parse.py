import layoutparser as lp
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# --- Step 1: Load the image using OpenCV (BGR format) ---
image_path = "/home/austere/Testing/ocrtesting/data/intermediate/fir2017/page_002.png"
image = cv2.imread(image_path)

# Optional: Convert to RGB for visualization (if needed)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Step 2: Load Detectron2 Layout Detection Model ---
model = lp.Detectron2LayoutModel(
    config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    extra_config=[
        "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.85,
        "MODEL.WEIGHTS", "/home/austere/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth"
    ],
)

# --- Step 3: Run layout detection ---
layout = model.detect(image)

# --- Step 4: Visualize layout blocks ---
# Show with Pillow, but if you prefer Matplotlib:
# plt.imshow(lp.draw_box(image_rgb.copy(), layout, box_width=3))
# plt.axis("off"); plt.show()
lp.draw_box(image_rgb.copy(), layout, box_width=3).show()

# --- Step 5: OCR using Tesseract ---
ocr_agent = lp.TesseractAgent(languages="eng+hin")
results = []

for block in layout:
    segment_image = block.crop_image(image)
    text = ocr_agent.detect(segment_image)['text']

    results.append({
        "type": block.type,
        "coordinates": block.coordinates,
        "text": text.strip()
    })

# --- Step 6: Print all results ---
for i, item in enumerate(results):
    print(f"\n---- Block {i+1} ----")
    print(f"Type        : {item['type']}")
    print(f"Coordinates : {item['coordinates']}")
    print("Text:")
    print(item['text'])
