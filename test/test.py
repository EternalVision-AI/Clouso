import fitz  # PyMuPDF

# Open the PDF file
pdf_document = "(1).pdf"
doc = fitz.open(pdf_document)

# Iterate through each page
for page_num in range(len(doc)):
    page = doc[page_num]
    print(f"Page {page_num + 1}:")

    # Get text blocks
    blocks = page.get_text("dict")["blocks"]

    # Iterate through text blocks
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span["font"]
                size = span["size"]
                color = span["color"]

                text = span["text"]
                print(f"  Text: {text}")
                print(f"    Font: {font}")
                print(f"    Size: {size}")
                print(f"    Color: {color:#x}")

doc.close()