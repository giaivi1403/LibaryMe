from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation , Flatten , Dense
from keras import backend as K

class LENET:
	@staticmethod
	def build(height , width , depth , classes):
		model = Sequential()

		input_shape = (height , width , depth)
		if K == "channel-first" :
			input_Shape = (depth , height , width)

		model.add(Conv2D(32,(5,5), padding = "same", input_shape = input_Shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size = (2,2) , strides = (2,2)))
		model.add(Conv2D(50,(5,5),padding = "same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size = (2,2) , strides = (2,2)))

		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model