# import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
import sounddevice
import soundfile
from scipy.io.wavfile import write
from kivy.config import Config
from kivy.uix.scrollview import ScrollView
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
import time
import librosa
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, Conv1D, MaxPooling1D, GlobalAveragePooling1D
import pandas as pd

Config.set('graphics', 'width', '1250')
Config.set('graphics', 'height', '500')

root = BoxLayout(orientation='vertical')
top_label = Label(text = 'Welcome to my humble Spoken Language Identificator', size=(1250, 50), size_hint=(None, None))
grid = GridLayout(cols=5)
instructions = Label(text = 'Instruction:\n\n 1) record your audio for 4 seconds \n(once you click to record it will start) \n\n2) play back your audio \n(you can play back the audio \nyou just recorded so you \nare confortable with it)\n\n3) test \n(you can test your audio by \nrunning it through the model',size=(250, 400))
record = Button(text = "Record",size=(250, 400),background_color=[0.5, 0.8, 1, 1], size_hint=(None, None))
play = Button(text = 'Play',size=(250, 400),background_color=[0.3, 0.5, 1, 1], size_hint=(None, None))
test = Button(text = 'Test',size=(250, 400),background_color=[0, 0.2, 1, 1], size_hint=(None, None))
result = Label(text = 'latest result:\n', size=(250, 400), size_hint=(None, None))
statusbar = Label(text = 'Status Bar!', size=(1250, 50), size_hint=(None, None))




class SLI_short_for_Spoken_Language_Identificator(App):
    def build(self):
        root.add_widget(top_label)
        grid.add_widget(instructions)

        
        record.bind(on_press = recorder)
        grid.add_widget(record)
        play.bind(on_press = player)
        grid.add_widget(play)
        play.disabled = True

        test.bind(on_press = tester)
        grid.add_widget(test)
        test.disabled = True
        grid.add_widget(result)
        root.add_widget(grid)
        root.add_widget(statusbar)

        return root


def recorder(self):
    statusbar.text = 'RECORDING, Please Speak in Your Microphone.'
    button_disabler(1)
    button_disabler(2)
    rec = sounddevice.rec(int(4 * 44100), samplerate=44100, channels=2)
    sounddevice.wait()
    write('rectest.wav', 44100, rec)
    print('new audio')
    button_enabler(1)
    button_enabler(2)
    statusbar.text = 'Latest Audio Ready to Be Played or Tested.'


def player(self):
    statusbar.text = 'PLAYING BACK, Please Make Sure You Are Satisfied With the Recording.'
    button_disabler(0)
    button_disabler(2)
    data, fs = soundfile.read('rectest.wav', dtype='float32')
    sounddevice.play(data, fs)
    time.sleep(4)
    button_enabler(0)
    button_enabler(2)
    statusbar.text = 'If You are Satisfied With the Recording You May Test it. If Not You Can Record Again.'

def tester(self):
    statusbar.text = 'TESTING, Please Be Patient.'
    button_disabler(0)
    button_disabler(1)
    data = extract_features('rectest.wav')
    features = [[data]]
    featuresdf = pd.DataFrame(features, columns=['feature'])
    X = np.array(featuresdf.feature.tolist())
    X = X.reshape(X.shape[0], 40, 1)
    model_result = model(X)
    model_result = result_processor(model_result)
    button_enabler(0)
    button_enabler(1)
    result.text = 'latest result:\n' + model_result
    statusbar.text = 'The Answer is Ready, You Can See it in the Result Section.'

def extract_features(file_name):

    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T,axis=0)
    
    return mfccsscaled

def model(data):

    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=2, input_shape=(40, 1), activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalAveragePooling1D())

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.load_weights('WaveModel.h5')

    score = model.predict(data)
    
    return score

def result_processor(model_result):
    maxind = 0
    for i in model_result:
        maxi = max(i)
        for j in range(3):
            if maxi == i[j]:
                maxind = j
    lang = 'ENGLISH'
    if maxind == 1:
        lang = 'CHINESE'
    if maxind == 2:
        lang = 'RUSSIAN'
    return lang

def button_disabler(button_num):

    if button_num == 0:
        record.disabled = True
        print('record should be disabled')
    elif button_num == 1:
        play.disabled = True
    else:
        test.disabled = True

def button_enabler(button_num):
    if button_num == 0:
        record.disabled = False
        print('record should be enabled')
    elif button_num == 1:
        play.disabled = False
    else:
        test.disabled = False

if __name__ == "__main__":
    SLI_short_for_Spoken_Language_Identificator().run()
    
