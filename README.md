# ocrtesting
Creating a custom hybrid pipeline to test out OCR strategies. Aim is to come up with a pipeline that outputs clean data for LLM model trainings.

hybrid_ocr_pipeline/

input/              ← Drop 2–3 FIR PDFs here

output/             ← JSON files will be saved here

modules/

    preprocess.py            ← PDF → image(s)

    ocr_paddle.py            ← PaddleOCR (hi+en)

    ocr_trocr.py             ← TrOCR (for handwritten)

    layout_parser.py         ← Detect tables, form sections

    field_parser.py          ← Regex + IndicNER for key fields


main.py                      ← orchestrates pipeline



## Pipeline FLow:

1. Convert PDF to Images-> Use 'pdf2image' or 'PyMuDf'
2. LayoutDetection-> Use 'LayoutParser': Identifies and extracts blocks(table, paragraph, handwriting)
3. OCR per Block-> 'PaddleOCR' for bilingual printed zones. 'PLATTER' for handwritten zones. 'Camelot' or 'pdfplumber' for tabular data.
4. Structured Output Builder-> Compose 'JSON' respone as output preserving block order by boudning box 'y' coordinate.


project_root/
├── main.py                          # Main runner script
├── requirements.txt                # All dependencies (PaddleOCR, pdf2image, etc.)
├── config.py                       # Global constants & paths
├── utils/
│   ├── file_utils.py               # PDF to image conversion
│   └── json_utils.py               # Output formatting utilities
├── modules/
│   ├── layout_detection.py         # LayoutParser integration
│   ├── ocr_paddle.py               # PaddleOCR handler (hi+en)
│   ├── ocr_platter.py              # PLATTER HTR handler for handwriting
│   ├── table_extraction.py         # Camelot/pdfplumber module
├── data/
│   ├── input/                      # Test PDF/image files
│   └── output/                     # Final JSON/structured results
├── pretrained/
│   └── platter_models/            # Pretrained weights for PLATTER (HTR)
└── README.md                       # Project setup + usage instructions
