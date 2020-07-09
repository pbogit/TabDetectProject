import random
import sys
import threading

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QCoreApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QComboBox

from TabDetect import TabDetect

class TabUI:

    def __init__(self):
        self.tabdetect = TabDetect()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.handleNewData)
        self.app = QApplication(sys.argv)

        self.window = QWidget()
        self.window.setGeometry(100, 100, 380, 190)
        self.window.setWindowTitle("TabDetect")
        self.window.setWindowIcon(QIcon("icon.png"))

        self.audioDevice = QComboBox()
        self.audioDevice.addItems(self.tabdetect.input_devices)
        self.startButton = QPushButton("Start")
        self.startButton.clicked.connect(self.startProcessing)

        self.tabLabels = QWidget()
        self.tabLabelLayout = QHBoxLayout()
        self.tabLabels.setLayout(self.tabLabelLayout)

        self.topSection = QWidget()
        self.topSectionLayout = QHBoxLayout()
        self.topSection.setLayout(self.topSectionLayout)
        self.topSectionLayout.addWidget(self.audioDevice, alignment=Qt.AlignTop)
        self.topSectionLayout.addWidget(self.startButton, alignment=Qt.AlignTop)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.topSection)
        self.layout.addWidget(self.tabLabels, alignment=Qt.AlignRight)

        self.window.setLayout(self.layout)

    def startProcessing(self):
        self.tabdetect.openStream(self.audioDevice.currentIndex())
        self.timer.start(5)

    def handleNewData(self):
        self.tabdetect.process()
        tabs = self.tabdetect.curr_tabs
        label = QLabel(tabs)
        self.tabLabelLayout.addWidget(label, alignment=Qt.AlignRight)
        if len(self.tabLabelLayout) > 35:
            item = self.tabLabelLayout.itemAt(0)
            self.tabLabelLayout.removeItem(item)
            item.widget().hide()

if __name__ == '__main__':
    tabUI = TabUI()
    tabUI.window.show()
    sys.exit(tabUI.app.exec_())



