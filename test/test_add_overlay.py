import fitz  # PyMuPDF
import fitz  # PyMuPDF
import cv2
import numpy as np

def add_searchable_text_layer(input_pdf, output_pdf, text, left=248, bottom=66):
    """
    Adds an additional searchable, invisible text layer to a PDF/A document.
    
    Parameters:
    input_pdf (str): Path to the input PDF file.
    output_pdf (str): Path to save the output PDF file with the text layer.
    """
    # Open the PDF file
    pdf_document = fitz.open(input_pdf)
    # extracted_data = extract_text_and_bboxes(input_pdf)
    
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        # Get the width and height of the current page
        width = page.rect.width
        height = page.rect.height
        print(width, height)
        # Render the page to a pixmap (image)
        pix = page.get_pixmap()

        # Convert the pixmap to a NumPy array
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Convert the image to BGR format if necessary (OpenCV uses BGR by default)
        if pix.n == 4:  # RGBA format
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        print("IMage", w, h)
        # Position where the text should be inserted (e.g., top-left corner)
        position = (left, bottom)  # Change according to your needs 225
        text = str(text)
        # left, top, right, bottom
        page.insert_text((370, 225), text, fontsize=12, set_simple=True, render_mode=3)
        page.insert_text((320, 350), '2024', fontsize=12, set_simple=True, render_mode=3)

    # Save the new PDF with the added text layer
    pdf_document.save(output_pdf, garbage=4, deflate=True, clean=True)

# Example usage
text_layers = ['0 1 0']
add_searchable_text_layer('(1).pdf', 'w_pdf_with_text.pdf', text_layers)
