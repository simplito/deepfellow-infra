"""EasyOCR service."""

from server.ocr import OcrOptions, ocr

service = ocr(OcrOptions(name="EasyOCR", language="en"))
