import cv2
import os
import numpy as np
from PIL import Image


from paddleocr import PaddleOCR
from classifier import Box_Classifier
from pdf2image import convert_from_path
from PIL import Image


# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

recognition_classes = ['chklist', 'year']

confThreshold = 0.5 # Confidence threshold
nmsThreshold = 0.9 # Non-maximum suppression threshold
dir_path = os.path.dirname(os.path.realpath(__file__))
detection_model = cv2.dnn.readNetFromONNX(dir_path + "./models/checkbox_detector_model.onnx")


def list_files(path):
	onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.pdf')]
	onlyfiles.sort()
	return onlyfiles


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

chkboxes = []
def DetectChecklist(img, options):
	global chkboxes
	detections = DetectionProcess(img)
	detected_cards = []
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
			chkboxes = Box_Classifier(detection_area)
		if detection['class_name'] == 'year' and 'year' in options:
			year_str = str(recognize_text_from_image(detection_area))
			cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 255), thickness=cv2.FILLED)
			cv2.putText(img, year_str, (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)

		detected_cards.append([left, top, right, bottom])
	h, w = img.shape[:2]



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
		pdf_path = inputFilename

		# Convert PDF to images
		images = convert_from_path(pdf_path)
		open_cv_images = []
		# Save each page as an image file
		for i, img in enumerate(images):
			# Convert the PIL image to a NumPy array
			open_cv_image = np.array(img)
			# Convert RGB to BGR (since OpenCV uses BGR format)
			open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
			open_cv_images.append(open_cv_image)
			img = DetectChecklist(open_cv_image, options)	
		images_to_pdf(open_cv_images, os.path.join(input_dir, input_file))
	return True


def preprocesser_file(input_file, options):
	# Convert PDF to images
 
	Image.MAX_IMAGE_PIXELS = None
	images = convert_from_path(input_file)
	open_cv_images = []
	# Save each page as an image file
	for i, img in enumerate(images):
		# Convert the PIL image to a NumPy array
		open_cv_image = np.array(img)
		# Convert RGB to BGR (since OpenCV uses BGR format)
		open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
		open_cv_images.append(open_cv_image)
		img = DetectChecklist(open_cv_image, options)	
	images_to_pdf(open_cv_images, input_file)
	return True



