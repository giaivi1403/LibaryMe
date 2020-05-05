import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
from keras.models import Sequential 
from keras.layers import Dense,Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


class CIFAR10:
	@staticmethod
	def build(height , width ,depth , classes):

		input_shape = (height , width , depth)
		if K.image_data_format() == "channel-first" :
			input_shape = (depth,height,width)
		
		model = Sequential()
		model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
		model.add(Activation('relu'))
		model.add(Conv2D(32, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dense(classes))
		model.add(Activation('softmax'))

		return model



