from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image

# Image path
image_path = './tempPDF.jpg'
pdf_path = 'output.pdf'

# Open the image to get dimensions
image = Image.open(image_path)
width, height = image.size

# Create a canvas
c = canvas.Canvas(pdf_path, pagesize=(width, height))

# Draw the image onto the canvas
c.drawImage(image_path, 0, 0, width, height)

# Save the PDF
c.save()

print(f"PDF saved as {pdf_path}")
