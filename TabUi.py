import os
import sys

import fretboardgtr
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QIcon
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, \
    QFileDialog, QLineEdit, QMessageBox
import pyqtgraph as pg
from fretboardgtr import ScaleGtr
from matplotlib import cm
import numpy as np

from TabDetect import TabDetect


class TabUI:

    def __init__(self):
        self.tabdetect = TabDetect(self)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateUi)
        self.app = QApplication(sys.argv)
        self.threadPool = QThreadPool()

        self.window = QWidget()
        self.window.setGeometry(100, 100, 1000, 370)
        self.window.setFixedSize(1000, 350)
        self.window.setWindowTitle("TabDetect")
        self.window.setWindowIcon(QIcon("icon.png"))

        self.audioDevice = QComboBox()
        self.audioDevice.addItems(self.tabdetect.input_devices)
        self.audioDevice.addItem("Audio file")
        self.startButton = QPushButton("Start")
        self.startButton.clicked.connect(self.processStartButton)

        self.modelFileButton = QPushButton("Select model")
        self.modelFileButton.clicked.connect(self.selectModel)
        self.modelFileLabel = QLineEdit("No model file selected")
        self.modelFileWidget = QWidget()
        self.modelFileWidget.setFixedSize(450, 20)
        self.modelFileLayout = QHBoxLayout()
        self.modelFileLayout.setContentsMargins(0, 0, 0, 0)
        self.modelFileWidget.setLayout(self.modelFileLayout)
        self.modelFileLayout.addWidget(self.modelFileButton)
        self.modelFileLayout.addWidget(self.modelFileLabel)

        self.audioFileButton = QPushButton("Select audio")
        self.audioFileButton.clicked.connect(self.selectAudio)
        self.audioFileLabel = QLineEdit("No audio file selected")
        self.audioFileWidget = QWidget()
        self.audioFileWidget.setFixedSize(450, 20)
        self.audioFileLayout = QHBoxLayout()
        self.audioFileLayout.setContentsMargins(0, 0, 0, 0)
        self.audioFileWidget.setLayout(self.audioFileLayout)
        self.audioFileLayout.addWidget(self.audioFileButton)
        self.audioFileLayout.addWidget(self.audioFileLabel)

        self.topSection = QWidget()
        self.topSection.setFixedSize(450, 20)
        self.topSectionLayout = QHBoxLayout()
        self.topSectionLayout.setContentsMargins(0, 0, 0, 0)
        self.topSection.setLayout(self.topSectionLayout)
        self.topSectionLayout.addWidget(self.audioDevice)
        self.topSectionLayout.addWidget(self.startButton)

        self.tabLabels = QWidget()
        self.tabLabels.setMinimumHeight(100)
        self.tabLabelLayout = QHBoxLayout()
        self.tabLabelLayout.setSpacing(0)
        self.tabLabelLayout.setContentsMargins(0, 0, 0, 0)
        self.tabLabels.setLayout(self.tabLabelLayout)

        self.fretboardWidget = QSvgWidget()
        self.fretboardPath = 'tmp/tabs.svg'
        self.fretboard = ScaleGtr()
        self.fretboard.customtuning(['E', 'A', 'D', 'G', 'B', 'E'])
        self.fretboard.theme(show_note_name=True, color_scale=False, last_fret=19)
        self.fretboard.pathname(self.fretboardPath)
        self.fretboard.draw(fingering=[None, None, None, None, None, None])
        self.fretboard.save()
        self.fretboardWidget.load(self.fretboardPath)
        os.remove(self.fretboardPath)

        self.tabHeatWidget = pg.PlotWidget()
        self.tabHeatWidget.getPlotItem().setTitle('Constant-Q Transform')
        self.tabHeatWidget.getPlotItem().hideAxis('bottom')
        self.tabHeatWidget.getPlotItem().hideAxis('left')
        self.tabHeatMap = pg.ImageItem()
        colormap = cm.get_cmap("plasma")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        self.tabHeatMap.setLookupTable(lut)
        self.tabHeatWidget.addItem(self.tabHeatMap)
        self.tabHeatMap.setImage(self.tabdetect.specs)

        self.leftWidget = QWidget()
        self.leftWidget.setFixedWidth(500)
        self.leftLayout = QVBoxLayout()
        self.leftLayout.setSpacing(5)
        self.leftLayout.addWidget(self.topSection)
        self.leftLayout.addWidget(self.audioFileWidget)
        self.leftLayout.addWidget(self.modelFileWidget)
        self.leftLayout.addWidget(self.tabLabels, alignment=Qt.AlignRight)
        self.leftLayout.addWidget(self.fretboardWidget)
        self.leftWidget.setLayout(self.leftLayout)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.leftWidget)
        self.layout.addWidget(self.tabHeatWidget)

        self.window.setLayout(self.layout)

    def selectModel(self):
        filePicker = QFileDialog()
        filePicker.setFileMode(QFileDialog.ExistingFile)
        filePicker.setNameFilter("Model files (*.h5)")
        filePicker.setDirectory("./")
        if filePicker.exec():
            self.modelFile = filePicker.selectedFiles()[0]
            self.modelFileLabel.setText(self.modelFile)

    def selectAudio(self):
        filePicker = QFileDialog()
        filePicker.setFileMode(QFileDialog.ExistingFile)
        filePicker.setNameFilter("Audio files (*.wav)")
        filePicker.setDirectory("./")
        if filePicker.exec():
            self.audioFile = filePicker.selectedFiles()[0]
            self.audioFileLabel.setText(self.audioFile)

    def processStartButton(self):
        if self.startButton.text() == "Start":
            try:
                self.tabdetect.init_model(self.modelFile)
                if self.audioDevice.currentText() != "Audio file":
                    self.tabdetect.openStream(self.audioDevice.currentIndex())
                else:
                    self.tabdetect.openFileStream(self.audioFile)
                self.timer.start(200)
                self.startButton.setText("Stop")
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Please select a valid model file.")
                msg.setDetailedText("The model file was either not set or was invalid. Please select a valid model "
                                    "file before clicking the \"Start\" button.")
                msg.setWindowTitle("No model found")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setMinimumWidth(500)
                msg.exec_()
        else:
            self.timer.stop()
            self.tabdetect.closeStream()
            self.tabdetect.wave_pos = 0
            for i in reversed(range(self.tabLabelLayout.count())):
                self.tabLabelLayout.itemAt(i).widget().setParent(None)
            self.fretboard.draw(fingering=[None, None, None, None, None, None])
            self.fretboard.save()
            self.fretboardWidget.load(self.fretboardPath)
            os.remove(self.fretboardPath)
            self.tabdetect.specs = np.zeros((25, 192))
            self.tabHeatMap.setImage(self.tabdetect.specs)
            self.startButton.setText("Start")

    def updateUi(self):
        tabs = self.tabdetect.curr_tabs
        label = QLabel(tabs)
        label.setFixedSize(18, 120)
        self.tabLabelLayout.addWidget(label, alignment=Qt.AlignRight)
        if len(self.tabLabelLayout) > 25:
            self.tabLabelLayout.itemAt(0).widget().setParent(None)
        self.fretboard.draw(fingering=self.tabdetect.curr_frets)
        self.fretboard.save()
        self.fretboardWidget.load(self.fretboardPath)
        os.remove(self.fretboardPath)
        self.tabHeatMap.setImage(self.tabdetect.specs)

if __name__ == '__main__':
    tabUI = TabUI()
    tabUI.window.show()
    sys.exit(tabUI.app.exec_())



