import layoutparser as lp
import cv2
from typing import List, Dict

def detect_layout_blocks(image_paths: List[str]) -> Dict[str, List[Dict]]:
    model = lp.PaddleDetectionLayoutModel(
        model_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/model",
        config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        enforce_cpu=True
    )

    results = {}
    for image_path in image_paths:
        image = cv2.imread(image_path)
        layout = model.detect(image)

        blocks = []
        for block in layout:
            x1, y1, x2, y2 = map(int, block.block.coordinates)
            blocks.append({
                "type": block.type,
                "bbox": [x1, y1, x2, y2],
                "score": float(block.score)
            })

        results[image_path] = blocks

    return results
