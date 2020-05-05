import numpy as np
import cv2
import os

class SimpleDatasetLoader : 
	def __init__ (self,preprocessors = None) :
		self.preprocessors = preprocessors

		if self.preprocessors is None :
			self.preprocessors = []


	def load(self,imagePaths,verbose = 1):
		data = []
		labels = []

		for(i , imagePath) in enumerate(imagePaths) :
			image = cv2.imread(imagePath)
			image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			label = imagePath.split(os.path.sep)[-1].split(".")[0]

			if self.preprocessors is not None :
				for p in self.preprocessors :
					img = p.preprocess(image)

			data.append(img)
			labels.append(label)

			if verbose > 0 and i > 0 and (i+1) % verbose == 0 :
				print("[INFO] processed {}/{}".format((i+1),len(imagePaths)))

		return (np.array(data),np.array(labels))


			
