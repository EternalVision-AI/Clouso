import cv2
import os
import numpy as np\
	

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

recognition_classes = ['chk', 'nochk']

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.6 # Non-maximum suppression threshold
dir_path = os.path.dirname(os.path.realpath(__file__))
detection_model = cv2.dnn.readNetFromONNX(dir_path + "./models/checkbox_classifier_model.onnx")


def DetectionProcess(original_image):
	[height, width, _] = original_image.shape
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


def Box_Classifier(img):
	detections = DetectionProcess(img)
	detected_areas = []
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
		
		detected_areas.append([(left, top, right, bottom), detection['class_name']])
		if detection['class_name'] == 'chk': 
			# Create a white rectangle on a copy of the image
			cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 255), thickness=cv2.FILLED)
			# Draw the red border around the white rectangle
			# cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), thickness=2)
			cv2.putText(img, "chx", (left+5, bottom-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)
		else: 
			# Create a white rectangle on a copy of the image
			cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 255), thickness=cv2.FILLED)
			# Draw the red border around the white rectangle
			# cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), thickness=2)
			cv2.putText(img, "cho", (left+5, bottom-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)

	# cv2.imshow("Test", cv2.resize(img, (int(w*2), int(h*2))))
	# cv2.waitKey(0)
	# Sort the detected_areas by the 'left' value of the bounding boxes
	detected_areas.sort(key=lambda x: x[0][0], reverse=True)  # Sort by the 'left' value (first element of the bounding box tuple)
	print(detected_areas)
	return detected_areas






