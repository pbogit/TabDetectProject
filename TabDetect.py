import time
import wave

import keras
import librosa
import pyaudio
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Activation
from keras import backend as K


class TabDetect:

    def __init__(self,
                 chunk_size=512,
                 sample_rate=44100,
                 ):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sample_rate_original = sample_rate
        self.sample_rate_downs = 22050
        self.num_strings = 6
        self.num_classes = 21
        self.input_shape = (192, 9, 1)
        self.specs = np.zeros((25, 192))
        self.p = pyaudio.PyAudio()
        self.count=0
        self.isFileStream = False
        self.waveFile = None
        self.curr_tabs = ''
        self.curr_frets = [None, None, None, None, None, None]
        self.spec = None
        self.stream = None
        self.model = None
        info = self.p.get_host_api_info_by_index(0)
        self.input_devices = []
        for i in range(0, info.get('deviceCount')):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                self.input_devices.append(self.p.get_device_info_by_host_api_device_index(0, i).get("name"))

    def openStream(self, index):
        self.isFileStream = False
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate_original,
                                  input=True, frames_per_buffer=self.chunk_size, input_device_index=index)

    def openFileStream(self, path):
        self.isFileStream = True
        self.waveFile = wave.open(path, 'rb')
        self.stream = self.p.open(format=self.p.get_format_from_width(self.waveFile.getsampwidth()), channels=self.waveFile.getnchannels(),
                                  rate=self.waveFile.getframerate(), output=True)

    def closeStream(self):
        self.stream.close()

    def preprocess_audio(self, data):
        data = data.astype(float)
        data = librosa.util.normalize(data)
        data = librosa.resample(data, self.sample_rate, self.sample_rate_downs)
        data = np.abs(librosa.cqt(data, hop_length=self.chunk_size, sr=self.sample_rate_downs, n_bins=192,
                                  bins_per_octave=24))
        return data

    def softmax_by_string(self, t):
        sh = K.shape(t)
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(K.expand_dims(K.softmax(t[:, i, :]), axis=1))
        return K.concatenate(string_sm, axis=1)

    def catcross_by_string(self, target, output):
        loss = 0
        for i in range(self.num_strings):
            loss += K.categorical_crossentropy(target[:, i, :], output[:, i, :])
        return loss

    def avg_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

    def formatTabs(self, tab):
        tabStr = ""
        fretBoard = []
        for string_num in range(len(tab)):
            fret_vector = tab[string_num]
            fret_class = np.argmax(fret_vector, -1)
            # 0 means that the string is closed
            if fret_class > 0:
                fret_num = fret_class - 1
                tabStr = str(fret_num) + "\n" + tabStr
                fretBoard.append(fret_num)
            else:
                tabStr = "-\n" + tabStr
                fretBoard.append(None)
        return tabStr, fretBoard

    def init_model(self, path):
        self.model = keras.models.load_model(path,
                                             custom_objects={'catcross_by_string': self.catcross_by_string,
                                                             'softmax_by_string': self.softmax_by_string,
                                                             'avg_acc': self.avg_acc})

    def processInput(self):
        data = np.frombuffer(self.stream.read(self.chunk_size * 17, exception_on_overflow=False), dtype=np.int16)
        self.applyModel(data)

    def processFile(self):
        originalData = self.waveFile.readframes(self.chunk_size * 8)
        while originalData != '':
            self.stream.write(originalData)
            data = np.frombuffer(originalData, dtype=np.int16)
            self.applyModel(data)
            originalData = self.waveFile.readframes(self.chunk_size * 8)

    def applyModel(self, data):
        self.spec = self.preprocess_audio(data)
        self.specs = np.append(self.specs, self.spec[:, 4].reshape((1, 192)), axis=0)
        self.specs = np.delete(self.specs, 0, axis=0)
        classes = self.model.predict(self.spec.reshape((1, 192, 9, 1)))[0]
        self.curr_tabs, self.curr_frets = self.formatTabs(classes)