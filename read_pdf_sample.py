import sys
try:
    import PyPDF2
    with open("data/sample.pdf", "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = reader.pages[0].extract_text()
        print("--- PAGE 1 ---")
        print(text[:1000])
except Exception as e:
    print(f"Failed with PyPDF2: {e}")
    try:
        import fitz
        doc = fitz.open("data/sample.pdf")
        print("--- PAGE 1 ---")
        print(doc[0].get_text()[:1000])
    except Exception as e2:
        print(f"Failed with fitz: {e2}")
