import fitz  # PyMuPDF
from PIL import Image

def add_searchable_text_layer(input_pdf, output_pdf, text):
    """
    Adds an additional searchable, invisible text layer to a PDF/A document.
    
    Parameters:
    input_pdf (str): Path to the input PDF file.
    output_pdf (str): Path to save the output PDF file with the text layer.
    """
    # Open the PDF file
    pdf_document = fitz.open(input_pdf)
    
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)

        # Position where the text should be inserted (e.g., top-left corner)
        # (left, top, right, bottom)
        # [[(364, 52, 408, 96), 'nochk'], [(219, 51, 262, 96), 'chk'], [(72, 51, 117, 96), 'nochk']]
        position = (350, 255)  # Change according to your needs
        
        # Font size and color
        fontsize = 8
        color = (0, 0, 0)  # Black color for visibility
        
        # 1. Insert the visible text layer
        # page.insert_text(position, text, fontsize=fontsize, color=color, render_mode=0)

        # 2. Insert the invisible text layer for searchability
        page.insert_text(position, text, fontsize=fontsize, color=(1, 1, 1, 0), render_mode=3)

    # Save the new PDF with the added text layer
    pdf_document.save(output_pdf, garbage=4, deflate=True, clean=True)

# Example usage
text_layers = ['0 1 0']
add_searchable_text_layer('w.pdf', 'w_pdf_with_text.pdf', text_layers)
