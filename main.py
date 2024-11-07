import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QDesktopWidget
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from obj_detection import detect_objects

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def run(self):
        cap = cv2.VideoCapture(self.filename)
        if not cap.isOpened():
            print("Error opening video file")
            return
        while True:
            ret, cv_img = cap.read()
            if ret:
                try:
                    frame, _ = detect_objects(cv_img)
                    self.change_pixmap_signal.emit(frame)
                except Exception as e:
                    print(f"Error processing frame: {e}")
            else:
                break
        cap.release()
        print("Video processing completed.")

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Smoke and Fire Detection'
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        
        # Obtain screen resolution and set window to half screen size
        screen = QDesktopWidget().screenGeometry()
        self.window_width = screen.width() // 2
        self.window_height = screen.height() // 2
        self.setGeometry(0, 0, self.window_width, self.window_height)
        
        # Layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Image label
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.window_width, self.window_height)  # Set fixed size
        layout.addWidget(self.image_label)
        
        # Browse button
        btn_browse = QPushButton('Browse', self)
        btn_browse.clicked.connect(self.openFileNameDialog)
        layout.addWidget(btn_browse)
        
        # Quit button
        btn_quit = QPushButton('Quit', self)
        btn_quit.clicked.connect(self.close)
        layout.addWidget(btn_quit)

        
    @pyqtSlot()
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select a file", "", "All Files (*);;Video Files (*.mp4);;Image Files (*.jpg; *.jpeg; *.png)", options=options)
        if fileName:
            if fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
                cv_img = cv2.imread(fileName)
                frame, _ = detect_objects(cv_img)
                self.displayImage(frame)
            else:
                self.thread = VideoThread(fileName)
                self.thread.change_pixmap_signal.connect(self.displayImage)
                self.thread.start()
                
    def displayImage(self, img):
        # Resize the image to the QLabel's size
        resized_image = cv2.resize(img, (self.image_label.width(), self.image_label.height()), interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        self.image_label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
