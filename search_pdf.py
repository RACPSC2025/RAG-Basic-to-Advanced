import fitz
import sys
import os

pdf_path = os.path.join("data", "sample.pdf")
if not os.path.exists(pdf_path):
    print(f"Error: {pdf_path} not found")
    sys.exit(1)

query = "2.2.1.1.7"
found = False

try:
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text()
        if query in text:
            print(f"✅ Found '{query}' on PAGE {i+1}!")
            start = text.find(query)
            print(f"Context: {text[max(0, start-100):start+500]}...")
            found = True
            # Not breaking here to see if there are more occurrences
    if not found:
        print(f"❌ '{query}' NOT FOUND in {pdf_path}")
    doc.close()
except Exception as e:
    print(f"Extraction failed: {e}")
