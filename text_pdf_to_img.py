from pdf2image import convert_from_path


# Path to your PDF file
pdf_path = 'w.pdf'

# Convert PDF to images
images = convert_from_path(pdf_path)

# Save each page as an image file
for i, image in enumerate(images):
    image.save(f'page_{i+1}.png', 'PNG')