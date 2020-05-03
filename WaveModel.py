# Load various imports 
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

'''
def extract_features(file_name):
   
    # try:
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T,axis=0)
     
    return mfccsscaled
    
    



features = []


c = 0
for filename in os.listdir('C:/Users/alisa\Desktop\Ali\Westminster\cmpt/390/allwave'):
    path = 'C:/Users/alisa\Desktop\Ali\Westminster\cmpt/390/allwave/'+filename
    data = extract_features(path)
    label = 0
    if filename[:2] == 'zh':
        label = 1
    elif filename[:2] == 'ru':
        label = 2
    features.append([data, label])
    c+=1
    print(c, label)
# print(features.shape)
random.shuffle(features)
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
# print(featuresdf.)
# random.shuffle
# print(type(featuresdf))

X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

train_x = X[:7000]
train_y = y[:7000]

test_x = X[7000:]
test_y = y[7000:]

np.save('train_x',train_x)
np.save('train_y',train_y)
np.save('test_x',test_x)
np.save('test_y',test_y)


'''
X = np.load('train_x.npy')
y = np.load('train_y.npy')


le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
print(x_train)
num_rows = 40
num_columns = 1
num_channels = 1
x_train = x_train.reshape(x_train.shape[0], num_rows, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_channels)

np.save('xtests', x_test)
np.save('ytests', y_test)
filter_size = 2

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

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


num_epochs = 75
num_batch_size = 25


model.fit(x_train, y_train, batch_size=num_batch_size, validation_data=(x_test, y_test), epochs=num_epochs, verbose=1)

model.save_weights('WaveModel.h5')
