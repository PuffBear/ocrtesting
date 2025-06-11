import layoutparser as lp
import cv2
from layoutparser.visualization import draw_box

# Load image
image = cv2.imread("/home/austere/Testing/ocrtesting/data/intermediate/fir2017/page_002.png")
if image is None:
    raise FileNotFoundError("Image not loaded. Please check the path.")

# OCR with Tesseract (Hindi + English)
ocr_agent = lp.TesseractAgent(languages="eng+hin")

# Return layout-aware objects for visualization
ocr_results = ocr_agent.detect(image, return_response=False)

ocr_text = ocr_agent.detect(image, return_response=True)
print(ocr_text)