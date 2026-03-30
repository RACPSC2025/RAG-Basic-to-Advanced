import fitz
import os
import sys

pdf_path = "data/sample.pdf"
doc = fitz.open(pdf_path)
page = doc[5] # Page 6
print(f"--- CONTENT PAGE 6 ---\n{page.get_text()}\n--- END PAGE 6 ---")
doc.close()
