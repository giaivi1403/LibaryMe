from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dense, BatchNormalization, Dropout, AveragePooling2D
import keras.backend as K

class LENET_5:
	@staticmethod

	def build(height, width , depth, classes):
		Inputshape = (height, width, depth)
		chanDim = -1
		# if K.image_data_format == "channel-first":
		# 	Inputshape = (depth, height, width)
		# 	chanDim = 1
		model = Sequential()
		
		#first layer
		model.add(Conv2D(1, (1,1), strides = (2,2), input_shape = Inputshape))
		model.add(Conv2D(6, (5,5)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = chanDim))
		
		#second layer		
		model.add(AveragePooling2D(pool_size = (2,2)))
		
		#third layer
		model.add(Conv2D(16, (5,5)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = chanDim))
		
		#fourth layer
		model.add(AveragePooling2D(pool_size = (2,2)))
		model.add(Flatten())
		
		#fifth layer
		model.add(Dense(120))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.25))

		#sixth layer
		model.add(Dense(84))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		#output layer
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model









