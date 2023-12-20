import sys
import PyQt5
from GestureRecognition import GestureRecognition
from PyQt5.QtCore import Qt
from app import Ui_MainWindow
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer, QSize
from PyQt5.QtGui import QPixmap
import subprocess
import cv2

class VideoCaptureWidget(QtWidgets.QWidget):
    def __init__(self, label, parent=None):
        super(VideoCaptureWidget, self).__init__(parent)
        self.video_size = QSize(320, 240)
        self.image_label = label
        self.setup_ui()
        self.setup_camera()

    def setup_ui(self):
        """Initialize widgets."""
        self.image_label.setFixedSize(self.video_size)

        # Set the layout
        #layout = QtWidgets.QVBoxLayout()
        #layout.addWidget(self.image_label)
        #self.setLayout(layout)

    def setup_camera(self):
        """Set up the camera index."""
        self.capture = cv2.VideoCapture(0)  # Index 0 for the default camera
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        # Use QTimer to capture frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms

    def update_frame(self):
        """Capture frame from the webcam and update QLabel."""
        ret, frame = self.capture.read()
        if ret:
            print("running")
            # Convert the frame to Qt format
            recognizer = GestureRecognition()
            frame = recognizer.processGestures(frame)
            frame = self.convert_cv_qt(frame)
            # Show the frame in the QLabel
            self.image_label.setPixmap(frame)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_size.width(), self.video_size.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        """Release the camera when the application closes."""
        self.capture.release()

class GameWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, ui=None):
        super(GameWindow, self).__init__(parent)
        self.ui = ui
        self.setWindowTitle("Game Window")
        self.setGeometry(100, 100, 1024, 768)

class CameraWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, ui=None):
        super(CameraWindow, self).__init__(parent)
        self.ui = ui
        self.setWindowTitle("Camera Window")
        self.setGeometry(700, 100, 350, 200)

        # Create Label for Camera
        self.webcamLabel = QtWidgets.QLabel(self)
        self.webcamLabel.setEnabled(True)
        self.webcamLabel.setGeometry(QtCore.QRect(10, 10, self.width() - 20, self.height() - 20))
        self.webcamLabel.setText("")
        self.webcamLabel.setObjectName("webcamLabel")

        # Create VideoCaptureWidget instance
        self.webcam = VideoCaptureWidget(self.webcamLabel)
        self.show()

    def resizeEvent(self, event):
        # Update the size of webcamLabel when the main window is resized
        self.webcamLabel.setGeometry(QtCore.QRect(10, 10, self.width() - 20, self.height() - 20))
        event.accept()

    def showEvent(self, event):
        # Set the window flags to stay on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.show()
        event.accept()

class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)

        #self.webcam = VideoCaptureWidget(self.ui.webcamLabel)


        # Connect button clicks to show pages and highlight buttons
        self.setup_button(self.ui.Profile1, self.showPage2)
        self.setup_button(self.ui.gameSelectionBtn, self.showPage3)
        self.setup_button(self.ui.profileSettingsBtn, self.showPage4)
        self.setup_button(self.ui.cameraSettingsBtn, self.showPage5)
        self.setup_button(self.ui.controlsSettingsBtn, self.showPage6)
        self.setup_button(self.ui.keyboardBtn_4, self.showPage6)
        self.setup_button(self.ui.controllerBtn_4, self.showPage7)
        self.setup_button(self.ui.keyboardBtn_5, self.showPage6)
        self.setup_button(self.ui.controllerBtn_5, self.showPage7)
        self.setup_button(self.ui.logoutBtn, self.showPage1)
        self.setup_button(self.ui.trackmaniaBtn, self.showPage8)
        self.setup_button(self.ui.controlsSettingsBtn_3, self.showPage2)
        self.setup_button(self.ui.logoutBtn_2, self.showPage1)

        # Set initial page and highlight the corresponding button
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_1)
        self.highlight_button(self.ui.logoutBtn)

    def setup_button(self, button, callback):
        button.clicked.connect(callback)
        button.setAutoFillBackground(True)
        button.setStyleSheet("""
            QPushButton {
                color: rgb(255, 255, 255);
                border: none;
            }
            QPushButton:hover {
                background-color: rgb(153, 178, 208);
            }
        """)

    def highlight_button(self, button):
        # Reset styles for all buttons
        for btn in [self.ui.Profile1, self.ui.gameSelectionBtn, self.ui.profileSettingsBtn,
                    self.ui.cameraSettingsBtn, self.ui.controlsSettingsBtn,
                    self.ui.logoutBtn, self.ui.trackmaniaBtn, self.ui.controlsSettingsBtn_3,
                    self.ui.logoutBtn_2]:
            btn.setStyleSheet("""
                QPushButton {
                    color: rgb(255, 255, 255);
                    border: none;
                }
                QPushButton:hover {
                    background-color: rgb(153, 178, 208);
                }
            """)

        # Set style for the highlighted button
        button.setStyleSheet("""
            QPushButton {
                background-color: rgb(99, 122, 155);
                color: rgb(255, 255, 255);
                border: none;
            }
        """)

    def show(self):
        self.main_win.show()

    def showPage1(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_1)
        self.highlight_button(self.ui.logoutBtn)

    def showPage2(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_2)
        self.highlight_button(self.ui.Profile1)
        self.highlight_button(self.ui.controlsSettingsBtn_3)
        self.highlight_button(self.ui.gameSelectionBtn)  # Highlight gameSelectionBtn

    def showPage3(self):
        self.ui.stackedWidget_2.setCurrentWidget(self.ui.page_3)
        self.highlight_button(self.ui.gameSelectionBtn)

    def showPage4(self):
        self.ui.stackedWidget_2.setCurrentWidget(self.ui.page_4)
        self.highlight_button(self.ui.profileSettingsBtn)

    def showPage5(self):
        self.ui.stackedWidget_2.setCurrentWidget(self.ui.page_5)
        self.highlight_button(self.ui.cameraSettingsBtn)

    def showPage6(self):
        self.ui.stackedWidget_2.setCurrentWidget(self.ui.page_6)
        self.highlight_button(self.ui.controlsSettingsBtn)
        self.highlight_button(self.ui.keyboardBtn_4)
        self.ui.controllerBtn_4.setStyleSheet("color: rgb(96, 100, 106);\n"
            "background-color: rgb(255, 255, 255);\n"
            "border:none;")
        self.ui.controlsSettingsBtn.setStyleSheet("background-color: rgb(99, 122, 155);\n"
                "color: rgb(255, 255, 255);\n"
                "border: none;")

    def showPage7(self):
        self.ui.stackedWidget_2.setCurrentWidget(self.ui.page_7)
        self.highlight_button(self.ui.controllerBtn_5)
        self.ui.keyboardBtn_5.setStyleSheet("color: rgb(96, 100, 106);\n"
            "background-color: rgb(255, 255, 255);\n"
            "border:none;")
        self.ui.controlsSettingsBtn.setStyleSheet("background-color: rgb(99, 122, 155);\n"
                "color: rgb(255, 255, 255);\n"
                "border: none;")

    def showPage8(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_8)
        self.highlight_button(self.ui.trackmaniaBtn)
        self.ui.launchGameBtn.clicked.connect(self.load_game)
        self.ui.launchGameBtn.clicked.connect(self.launch_camera_window)

    def launch_game_window(self):
        game_window = GameWindow(self.main_win, self.ui)
        game_window.show()

    def launch_camera_window(self):
        camera_window = CameraWindow(self.main_win, self.ui)
        camera_window.show()

    def load_game(self):
        # Specify the path to the Trackmania executable
        trackmania_path = r"C:\Program Files (x86)\Steam\steamapps\common\Trackmania\Trackmania.exe"

        # Specify the desired screen resolution
        resolution = "1600x900"  # Change this to your desired resolution

        # Use subprocess to launch Trackmania with the specified resolution
        subprocess.Popen([trackmania_path, f"-screen-width {resolution.split('x')[0]}", f"-screen-height {resolution.split('x')[1]}"])
 

if __name__ == '__main__':
    # Enable High DPI display with PyQt5
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    main_win = MainWindow()

    main_win.show()
    sys.exit(app.exec_())
