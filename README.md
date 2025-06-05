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