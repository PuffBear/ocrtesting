import sys
import torch
import torchvision.transforms as T
from PIL import Image
import pytesseract 
from cascade_tabnet import CascadeTabNet

def load_model(checkpoint_path, device='cuda'):
    model = CascadeTabNet(backbone='resnet50', num_classes=2)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def detect_tables(model, img, device='cuda', score_thr=0.5):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outs = model(inp)[0]
    boxes = outs['boxes'].cpu()
    scores = outs['scores'].cpu()
    # filter by detection score
    keep = scores > score_thr
    return boxes[keep].numpy()

def ocr_table_crops(img, table_boxes):
    texts = []
    for idx, (x1,y1,x2,y2) in enumerate(table_boxes.astype(int)):
        crop = img.crop((x1, y1, x2, y2))
        # --psm 6 treats the crop as a uniform block of text
        txt = pytesseract.image_to_string(crop, config='--psm 6').strip()
        texts.append((idx, txt))
    return texts

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect_and_ocr.py <image.png>")
        sys.exit(1)

    # 1) load image
    img_path = sys.argv[1]
    img = Image.open(img_path).convert("RGB")

    # 2) load your CascadeTabNet model
    #    replace 'cascadetabnet_table.pth' with your checkpoint path
    model = load_model("cascadetabnet_table.pth")

    # 3) detect table boxes
    table_boxes = detect_tables(model, img)

    # 4) OCR each table
    results = ocr_table_crops(img, table_boxes)

    # 5) print out
    for idx, text in results:
        print(f"\n=== Table {idx} ===\n{text}\n")