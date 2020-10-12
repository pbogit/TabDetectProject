import sys

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, \
    QFileDialog, QLineEdit, QMessageBox
import pyqtgraph as pg
from matplotlib import cm
import numpy as np

from TabDetect import TabDetect


class TabUI:

    def __init__(self):
        self.tabdetect = TabDetect()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.handleNewData)
        self.app = QApplication(sys.argv)

        self.window = QWidget()
        self.window.setGeometry(100, 100, 1000, 350)
        self.window.setFixedSize(1000, 350)
        self.window.setWindowTitle("TabDetect")
        self.window.setWindowIcon(QIcon("icon.png"))

        self.audioDevice = QComboBox()
        self.audioDevice.addItems(self.tabdetect.input_devices)
        self.startButton = QPushButton("Start")
        self.startButton.clicked.connect(self.processStartButton)

        self.fileButton = QPushButton("Select model")
        self.fileButton.clicked.connect(self.selectModel)
        self.fileLabel = QLineEdit("No model selected")
        self.fileLabel.setReadOnly(True)
        self.filePicker = QWidget()        
        self.filePicker.setFixedSize(450, 20)
        self.filePickerLayout = QHBoxLayout()
        self.filePickerLayout.setContentsMargins(0, 0, 0, 0)
        self.filePicker.setLayout(self.filePickerLayout)
        self.filePickerLayout.addWidget(self.fileButton)
        self.filePickerLayout.addWidget(self.fileLabel)

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

        # TODO enter Fretboard visualization

        self.tabHeatWidget = pg.PlotWidget()
        self.tabHeatWidget.getPlotItem().setTitle('Constant-Q Transform')
        self.tabHeatWidget.getPlotItem().hideAxis('bottom')
        self.tabHeatWidget.getPlotItem().hideAxis('left')
        self.tabHeatMap = pg.ImageItem()
        colormap = cm.get_cmap("plasma")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        self.tabHeatMap.setLookupTable(lut)
        self.tabHeatWidget.addItem(self.tabHeatMap)
        self.tabHeatMap.setImage(self.tabdetect.specs)

        self.leftWidget = QWidget()
        self.leftWidget.setFixedWidth(500)
        self.leftLayout = QVBoxLayout()
        self.leftLayout.setSpacing(5)
        self.leftLayout.addWidget(self.filePicker)
        self.leftLayout.addWidget(self.topSection)
        self.leftLayout.addWidget(self.tabLabels, alignment=Qt.AlignRight)
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
            self.fileLabel.setText(self.modelFile)

    def processStartButton(self):
        if self.startButton.text() == "Start":
            try:
                self.tabdetect.init_model(self.modelFile)
                self.tabdetect.openStream(self.audioDevice.currentIndex())
                self.timer.start(100)
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
            for i in reversed(range(self.tabLabelLayout.count())):
                self.tabLabelLayout.itemAt(i).widget().setParent(None)
            self.startButton.setText("Start")


    def handleNewData(self):
        self.tabdetect.process()
        tabs = self.tabdetect.curr_tabs
        label = QLabel(tabs)
        label.setFixedSize(18, 120)
        self.tabLabelLayout.addWidget(label, alignment=Qt.AlignRight)
        if len(self.tabLabelLayout) > 25:
            self.tabLabelLayout.itemAt(0).widget().setParent(None)
        self.tabHeatMap.setImage(self.tabdetect.specs)

if __name__ == '__main__':
    tabUI = TabUI()
    tabUI.window.show()
    sys.exit(tabUI.app.exec_())



