from server.ocr import ocr, OcrOptions

service = ocr(OcrOptions(
    name="EasyOCR",
    language="en"
    ))
