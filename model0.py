import keras






model = keras.models.Sequential()
model.add(keras.layers.Conv2D(500, (39, 39), input_shape=(3, , )))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))



model.save_weights('MODEL0.h5')