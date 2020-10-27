import wave

import keras
import librosa
import pyaudio
import numpy as np
from keras import backend as K


class TabDetect:

    def __init__(self,
                 tabUi,
                 chunk_size=512,
                 ):
        self.tabUi = tabUi
        self.chunk_size = chunk_size
        self.sample_rate = 22050
        self.num_strings = 6
        self.num_classes = 21
        self.input_shape = (192, 9, 1)
        self.specs = np.zeros((25, 192))
        self.p = pyaudio.PyAudio()
        self.wave_pos = 0
        self.wave_len = 0
        self.isFileStream = False
        self.audioFile = None
        self.curr_tabs = ''
        self.curr_frets = [None, None, None, None, None, None]
        self.spec = None
        self.stream = None
        self.model = None
        self.textFile = None
        info = self.p.get_host_api_info_by_index(0)
        self.input_devices = []
        for i in range(0, info.get('deviceCount')):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                self.input_devices.append(self.p.get_device_info_by_host_api_device_index(0, i).get("name"))

    def openStream(self, index):
        self.isFileStream = False
        self.audioFile = wave.open('input_audio.wav', 'wb')
        self.audioFile.setnchannels(1)
        self.audioFile.setnframes(8192)
        self.audioFile.setframerate(self.sample_rate)
        self.audioFile.setsampwidth(2)
        self.textFile = open("input_tabs.txt", "w")
        self.textFile.write('EADGBE\n')
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
                                  input=True, frames_per_buffer=4096, input_device_index=index,
                                  stream_callback=self.processInput)


    def openFileStream(self, path):
        self.isFileStream = True
        self.audioFile = librosa.load(path, sr=self.sample_rate, mono=True)
        self.wave_len = len(self.audioFile[0])
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1,
                                  rate=22050, output=True, stream_callback=self.processFile, frames_per_buffer=4096)

    def closeStream(self):
        if not self.isFileStream :
            self.textFile.close()
            self.audioFile.close()
        self.stream.close()

    def preprocess_audio(self, data):
        data = data.astype(float)
        data = librosa.util.normalize(data)
        data = np.abs(librosa.cqt(data, hop_length=self.chunk_size, sr=self.sample_rate, n_bins=192,
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


    def processInput(self, _data, frames, _time, status_flags):
        self.audioFile.writeframes(_data)
        data = np.frombuffer(_data, dtype=np.int16)
        self.applyModel(data)
        return data, pyaudio.paContinue

    def processFile(self, _data, frames, _time, status_flags):
        left = self.wave_pos
        right = min(self.wave_pos + frames, self.wave_len - 1)
        data = np.ndarray.tobytes(self.audioFile[0][left: right])
        self.applyModel(self.audioFile[0][left: right])
        self.wave_pos += frames
        if self.wave_pos >= self.wave_len:
            return data, pyaudio.paComplete
        return data, pyaudio.paContinue

    def readFrames(self):
        nframes = self.chunk_size * 18 - 2
        frames = self.audioFile.readframes(nframes)
        return frames

    def applyModel(self, data):
        self.spec = self.preprocess_audio(data)
        self.specs = np.append(self.specs, self.spec[:, 4].reshape((1, 192)), axis=0)
        self.specs = np.delete(self.specs, 0, axis=0)
        classes = self.model.predict(self.spec.reshape((1, 192, 9, 1)))[0]
        self.curr_tabs, self.curr_frets = self.formatTabs(classes)
        if not self.isFileStream:
            self.textFile.write(self.curr_tabs.replace('\n', '')[::-1])
            self.textFile.write('\n')
