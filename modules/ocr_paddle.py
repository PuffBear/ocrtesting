# modules/ocr_paddle.py

from paddleocr import PaddleOCR

def init_paddle_ocr():
    return PaddleOCR(use_angle_cls=True, lang='hi+en', show_log=False)

def run_ocr_on_image(ocr_model, image_path):
    result = ocr_model.ocr(image_path, cls=True)
    blocks = []
    for line in result[0]:
        box = [int(pt) for point in line[0] for pt in point]
        text = line[1][0]
        score = float(line[1][1])
        blocks.append({
            "bbox": box,
            "text": text,
            "score": score
        })
    return blocks
