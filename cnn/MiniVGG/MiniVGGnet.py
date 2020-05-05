from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation,Flatten,Dropout,Dense 
from keras import backend as K

class MINIVGGNET:
	@staticmethod
	def build(height, width, depth, classes):
		Input_shape = (height,width,depth)
		chanDim = -1
		if K.image_data_format == "channel-first":
			Input_shape = (depth,height,width)
			chanDim = 1
		model = Sequential()

		#first CONV => RELU => CONV => RELU => POOL layer
		model.add(Conv2D(32, (3,3), input_shape =Input_shape, padding = "same"))
		model.add(Activation("relu"))
		model.add(Conv2D(32, (3,3), padding = "same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = chanDim))
		model.add(MaxPooling2D(pool_size = (2,2)))
		model.add(Dropout(0.25))

		#second CONV => RELU => CONV => RELU => POOL layer
		model.add(Conv2D(64, (3,3), padding = "same"))
		model.add(Activation("relu"))
		model.add(Conv2D(64, (3,3), padding = "same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = chanDim))
		model.add(MaxPooling2D(pool_size = (2,2)))
		model.add(Dropout(0.25))

		#first FULL CONNECT layer set
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		#second FULL CONNECT layer set
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model