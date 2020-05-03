import pandas as pd
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
import random


x_test = np.load('test_x.npy')
y_test = np.load('test_y.npy')

num_rows = 40
num_channels = 1
x_test = x_test.reshape(x_test.shape[0], num_rows, num_channels)

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, padding='same', input_shape=(num_rows, num_channels), activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=16, kernel_size=2, padding='same', activation='relu'))

model.add(GlobalAveragePooling1D())

model.add(Dense(3, activation='softmax'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

model.load_weights('WaveModel.h5')

score = model.predict(x_test)
c = 0
compare = []
for i in score:
    maxi = max(i)
    maxind = 0
    for j in range(3):
        if maxi == i[j]:
            maxind = j
    compare.append([maxind,y_test[c]])

    c += 1

losses = 0
for i in compare:
    if i[0] != i[1]:
        losses +=1
acc = (1703-losses)*100/1703
print('the accuracy of the model is:',acc)