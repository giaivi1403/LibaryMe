import cv2

class SimplePreProcessor :
	def __init__ (self, height, width , inter = cv2.INTER_AREA):

		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self,image) :
		return cv2.resize(image,(self.width,self.height),interpolation = self.inter)


