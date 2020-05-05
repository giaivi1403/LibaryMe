import cv2
import numpy as np

class SimpleDatasetConvertGray:
	def __init__(self,data_format = cv2.COLOR_BGR2GRAY):
		self.data_format = data_format

	def preprocess(self,data_array) :
		gray_data = []
		for image in data_array:
			image = cv2.cvtColor(image,self.data_format)
			image = image.reshape(28,28,1)
			gray_data.append(image)

		return np.array(gray_data)