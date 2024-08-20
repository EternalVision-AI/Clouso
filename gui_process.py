import cv2
import os
import numpy as np
from PIL import Image


from paddleocr import PaddleOCR
from classifier import Box_Classifier
from pdf2image import convert_from_path
from PIL import Image
import fitz

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

recognition_classes = ['chklist', 'year']

confThreshold = 0.3 # Confidence threshold
nmsThreshold = 0.5 # Non-maximum suppression threshold
dir_path = os.path.dirname(os.path.realpath(__file__))
detection_model = cv2.dnn.readNetFromONNX(dir_path + "./models/checkbox_detector_model.onnx")


def list_files(path):
	onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.pdf')]
	onlyfiles.sort()
	return onlyfiles

def pdf_to_images(pdf_path):
    """
    Converts a PDF file into a list of images represented as NumPy arrays.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List[np.ndarray]: A list of images as NumPy arrays, each representing a page of the PDF.
    """
    # Load the PDF document
    pdf_document = fitz.open(pdf_path)

    # List to hold all the image arrays
    images = []

    # Iterate through each page in the document
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        
        # Render the page to a pixmap (image)
        pix = page.get_pixmap()

        # Convert the pixmap to a NumPy array
        image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Convert to BGR format if necessary (OpenCV uses BGR by default)
        if pix.n == 4:  # RGBA format
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB format
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Append the image array to the list
        images.append(image_array)

    # Close the PDF document
    pdf_document.close()

    return images
  
def recognize_text_from_image(img):
  	# Initialize PaddleOCR with English model
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can specify other languages if needed# Perform OCR on the image
    result = ocr.ocr(img, cls=True)
    # Extracting text and its corresponding bounding box coordinates
    recognized_text = []
    final_text = ""
    for line in result:
        for (bbox, text) in line:
            final_text = f"{final_text} {text[0]}"
            recognized_text.append({
                "text": text[0],
                "bounding_box": bbox
            })

    return final_text

def add_searchable_text_layer(input_pdf, output_pdf, add_text):
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

        for text in add_text:
          print(add_text)
          print(text)
					# Position where the text should be inserted (e.g., top-left corner)
          position = (text[0], text[1])  # Change according to your needs 225
					# left, top, right, bottom
          page.insert_text(position, text[2], fontsize=10, set_simple=True, render_mode=3)

    # Save the new PDF with the added text layer
    pdf_document.save(output_pdf, garbage=4, deflate=True, clean=True)

def DetectionProcess(original_image):
	[height, width] = original_image.shape[:2]
	length = max((height, width))
	image = np.zeros((length, length, 3), np.uint8)
	image[0:height, 0:width] = original_image
	scale = length / INPUT_WIDTH

	blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(INPUT_WIDTH, INPUT_WIDTH), swapRB=True)
	detection_model.setInput(blob)
	outputs = detection_model.forward()

	outputs = np.array([cv2.transpose(outputs[0])])
	rows = outputs.shape[1]

	boxes = []
	scores = []
	class_ids = []

	for i in range(rows):
		classes_scores = outputs[0][i][4:]
		(minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
		if maxScore >= confThreshold:
			box = [
				outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
				outputs[0][i][2], outputs[0][i][3]]
			boxes.append(box)
			scores.append(maxScore)
			class_ids.append(maxClassIndex)

	result_boxes = cv2.dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold)

	detections = []
	for i in range(len(result_boxes)):
		index = result_boxes[i]
		box = boxes[index]
		detection = {
			'class_id': class_ids[index],
			'class_name': recognition_classes[class_ids[index]],
			'confidence': scores[index],
			'box': box,
			'scale': scale}
		detections.append(detection)
	return detections

def DetectChecklist(img, input_file, options):

	detections = DetectionProcess(img)
	detected_cards = []
	add_text = []
	for detection in detections:
		class_id, class_name, confidence, box, scale = \
			detection['class_id'], detection['class_name'], detection['confidence'], detection['box'], detection[
				'scale']
		left, top, right, bottom = round(box[0] * scale), round(box[1] * scale), round(
			(box[0] + box[2]) * scale), round((box[1] + box[3]) * scale)
		if left<0: left = 0
		if top<0: top = 0
		if right>img.shape[1]: right = img.shape[1] - 1
		if bottom>img.shape[0]: bottom = img.shape[0] - 1

		detection_area = img[top:bottom, left:right]
		if detection['class_name'] == 'chklist' and 'checkbox' in options:
			# cv2.putText(img, str((left, bottom)), (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)
			checkbox_text = Box_Classifier(detection_area)
			add_text.append([int((left+right)/2), bottom, checkbox_text])
		if detection['class_name'] == 'year' and 'year' in options:
			year_text = str(recognize_text_from_image(detection_area))
			add_text.append([int((left+right)/2), bottom, year_text])
	directory_path, filename = os.path.split(input_file)
	# Define the new directory path
	processed_directory = directory_path + "_boosted"

	# Create the new directory if it doesn't exist
	if not os.path.exists(processed_directory):
			os.makedirs(processed_directory)
	output_file = processed_directory+"/"+filename
 
	add_searchable_text_layer(input_file, output_file, add_text)

		# detected_cards.append([left, top, right, bottom])
	



def opencv_to_pil(cv_image):
    # Convert from OpenCV BGR format to RGB format for PIL
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

def images_to_pdf(image_files, output_pdf):
    # Convert all OpenCV images to PIL images
    pil_images = [opencv_to_pil(cv_image) for cv_image in image_files]

    # Save the first image and append the rest as additional pages
    pil_images[0].save(output_pdf, save_all=True, append_images=pil_images[1:])

def preprocesser_folder(input_dir, options):
	ls_files = list_files(input_dir)
	for input_file in ls_files:
		inputFilename = os.path.join(input_dir, input_file)
		# Path to your PDF file
		preprocesser_file(inputFilename, options)
	return True


def preprocesser_file(input_file, options):
	# Convert PDF to images
 
	Image.MAX_IMAGE_PIXELS = None
	images = pdf_to_images(input_file)
	# Save each page as an image file
	for i, open_cv_image in enumerate(images):
		DetectChecklist(open_cv_image, input_file, options)	
	# images_to_pdf(open_cv_images, input_file)
	return True



