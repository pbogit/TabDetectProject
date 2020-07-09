import time

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
        self.count = 0
        self.input_shape = (192, 9, 1)
        self.init_model()
        self.p = pyaudio.PyAudio()
        info = self.p.get_host_api_info_by_index(0)
        self.input_devices = []
        for i in range(0, info.get('deviceCount')):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                self.input_devices.append(self.p.get_device_info_by_host_api_device_index(0, i).get("name"))

    def openStream(self, index):

        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate_original,
                                  input=True, frames_per_buffer=self.chunk_size, input_device_index=index)

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

    def tab2str(self, tab):
        tabStr = ""
        for string_num in range(len(tab)):
            fret_vector = tab[string_num]
            fret_class = np.argmax(fret_vector, -1)
            # 0 means that the string is closed
            if fret_class > 0:
                fret_num = fret_class - 1
                tabStr = str(fret_num) + "\n" + tabStr
            else:
                tabStr = "-\n" + tabStr
        return tabStr

    def init_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes * self.num_strings))  # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string))
        model.compile(loss=self.catcross_by_string,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=[self.avg_acc])
        model.load_weights('weights.h5')
        self.model = model

    def process(self):
        data = np.frombuffer(self.stream.read(self.chunk_size * 17, exception_on_overflow=False), dtype=np.int16)
        spec = self.preprocess_audio(data)
        classes = self.model.predict(spec.reshape((1, 192, 9, 1)))[0]
        self.curr_tabs = self.tab2str(classes)
        self.count = 0
