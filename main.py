import sys
import PyQt5
from gestures import GestureRecognizer
from PyQt5.QtCore import Qt
from app import Ui_MainWindow
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QInputDialog
from PyQt5.QtCore import QTimer, QSize
from PyQt5.QtGui import QPixmap
import subprocess
import cv2
import json
import os

class VideoCaptureWidget(QtWidgets.QWidget):
    def __init__(self, label, parent=None):
        super(VideoCaptureWidget, self).__init__(parent)
        self.video_size = QSize(320, 240)
        self.image_label = label
        self.setup_ui()
        self.setup_camera()
        self.active = False

        # Connect the update_frame method to the timer
        self.timer.timeout.connect(self.update_frame)

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
            #print("running")
            # Convert the frame to Qt format
            recognizer = GestureRecognizer()
            frame = recognizer.RecognizeGestures(frame)
            frame = self.convert_cv_qt(frame)
            # Show the frame in the QLabel
            self.image_label.setPixmap(frame)

    def set_preview_label(self, label):
        self.image_label = label
        self.image_label.setFixedSize(self.video_size)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_size.width(), self.video_size.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def print_camera_info(self):
        """Print information about the camera."""
        camera_info = self.capture.getBackendName()
        actual_camera_name = self.capture.get(cv2.CAP_PROP_BACKEND)
        print(f"Camera backend: {camera_info}")
        print(f"Actual camera name: {actual_camera_name}")

    def start_camera(self):
        self.timer.start()
        self.active = True

    def stop_camera(self):
        self.timer.stop()
        self.active = False

        # Add a new method to pause the camera feed

    def pause_camera(self):
        self.timer.stop()
        self.active = False
        self.camera_paused = True

        # Add a new method to resume the camera feed

    def resume_camera(self):
        if not self.camera_paused:
            return

        self.timer.start()
        self.active = True
        self.camera_paused = False

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

class CreateGestureWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, ui=None):
        super(CreateGestureWindow, self).__init__(parent)
        self.ui = ui
        self.setWindowTitle("Gesture Window")
        self.setGeometry(100, 100, 350, 200)

        self.gesture_recognizer = GestureRecognizer()

        # Create Label for Camera
        self.webcamLabel = QtWidgets.QLabel(self)
        self.webcamLabel.setEnabled(True)
        self.webcamLabel.setGeometry(QtCore.QRect(10, 10, self.width() - 20, self.height() - 70))
        self.webcamLabel.setText("")
        self.webcamLabel.setObjectName("webcamLabel")

        # Create VideoCaptureWidget instance
        self.webcam = VideoCaptureWidget(self.webcamLabel)

        # Create Record Button
        self.recordButton = QPushButton("Record", self)
        self.recordButton.setGeometry(10, self.height() - 60, 100, 50)
        self.recordButton.clicked.connect(self.start_recording)

        # Create Countdown Label
        self.countdownLabel = QLabel(self)
        self.countdownLabel.setGeometry(120, self.height() - 60, 100, 50)
        self.countdownLabel.setAlignment(Qt.AlignCenter)
        self.countdownLabel.setStyleSheet("font-size: 18px;")

        # Variable to store countdown value
        self.countdown_value = 3

        # Initialize a list to store landmarks
        self.landmarks_list = []

        # Connect the destroyed signal to stop the camera when the window is closed
        self.destroyed.connect(self.stop_camera_on_gesture_window_close)

        # Connect the update_frame method to the timer
        self.webcam.timer.timeout.connect(self.update_frame)

    def start_recording(self):
        # Disable the record button during recording
        self.recordButton.setEnabled(False)

        # Start countdown
        self.start_countdown()

        # Clear existing landmarks
        self.landmarks_list.clear()

    def start_countdown(self):
        if self.countdown_value > 0:
            self.countdownLabel.setText(str(self.countdown_value))
            self.countdown_value -= 1
            QtCore.QTimer.singleShot(1000, self.start_countdown)
        else:
            self.countdownLabel.clear()
            # Start recording for 10 seconds
            QtCore.QTimer.singleShot(10000, self.stop_recording)

    def stop_recording(self):
        # Enable the record button after recording
        self.recordButton.setEnabled(True)

        # Stop the camera when recording is complete
        self.webcam.pause_camera()

        # Prompt the user to enter the gesture name using QMessageBox
        gesture_name, ok = QInputDialog.getText(self, 'Gesture Name', 'Enter the gesture name:')

        if ok and gesture_name:
            print("Landmarks recorded:")
            non_empty_landmarks = [landmarks for landmarks in self.landmarks_list if any(landmarks)]
            for landmarks in non_empty_landmarks:
                print(landmarks)

            print(f"Gesture recorded with name: {gesture_name}")

            # Specify the folder path
            folder_path = "CreatedGestures"

            # Create the folder if it doesn't exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Save non-empty landmarks to a file with the specified gesture name in the folder
            filename = os.path.join(folder_path, f"{gesture_name}.json")
            with open(filename, 'w') as file:
                json.dump(non_empty_landmarks, file)

            print(f"Non-empty landmarks saved to {filename}")

            if hasattr(self, 'webcam'):
                self.webcam.resume_camera()
        else:
            print("Please enter a gesture name.")

            if hasattr(self, 'webcam'):
                self.webcam.resume_camera()

    def stop_camera_on_gesture_window_close(self):
        # Stop the camera feed when the CreateGestureWindow is closed
        self.webcam.stop_camera()

    def update_frame(self):
        # Capture frame from the webcam and update QLabel
        ret, frame = self.webcam.capture.read()
        if ret:
            # Recognize gestures and get landmarks
            landmarks = self.gesture_recognizer.get_landmarks(frame)

            # Convert landmarks to pixel coordinates
            image_width, image_height = self.webcam.video_size.width(), self.webcam.video_size.height()
            landmark_coordinates = self.gesture_recognizer.get_landmarks(frame)

            # Append landmarks to the list
            self.landmarks_list.append(landmark_coordinates)

            # Convert the frame to Qt format and show in the QLabel
            frame = self.webcam.convert_cv_qt(frame)
            self.webcam.image_label.setPixmap(frame)

            # Update the frame in VideoCaptureWidget
            # self.webcam.update_frame()
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
        self.setup_button(self.ui.pushButton, self.showPage11)
        self.setup_button(self.ui.BtnTutorial, self.showPage10)
        self.setup_button(self.ui.AddProfileButton, self.showPage9)
        self.setup_button(self.ui.pushButton, self.showPage11)
        self.setup_button(self.ui.pushButton_2,self.showPage6)
        self.setup_button(self.ui.Back,self.showPage1)

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

        # stop camera feed
        if hasattr(self, 'webcam') and not self.ui.stackedWidget_2.currentWidget() == self.ui.page_5:
            self.webcam.stop_camera()

    def showPage2(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_2)
        self.highlight_button(self.ui.Profile1)
        self.highlight_button(self.ui.controlsSettingsBtn_3)
        self.highlight_button(self.ui.gameSelectionBtn)  # Highlight gameSelectionBtn

        # stop camera feed
        if hasattr(self, 'webcam') and not self.ui.stackedWidget_2.currentWidget() == self.ui.page_5:
            self.webcam.stop_camera()

    def showPage3(self):
        self.ui.stackedWidget_2.setCurrentWidget(self.ui.page_3)
        self.highlight_button(self.ui.gameSelectionBtn)

        # stop camera feed
        if hasattr(self, 'webcam') and not self.ui.stackedWidget_2.currentWidget() == self.ui.page_5:
            self.webcam.stop_camera()

    def showPage4(self):
        self.ui.stackedWidget_2.setCurrentWidget(self.ui.page_4)
        self.highlight_button(self.ui.profileSettingsBtn)

        # stop camera feed
        if hasattr(self, 'webcam') and not self.ui.stackedWidget_2.currentWidget() == self.ui.page_5:
            self.webcam.stop_camera()

    def showPage5(self):
        self.ui.stackedWidget_2.setCurrentWidget(self.ui.page_5)
        self.highlight_button(self.ui.cameraSettingsBtn)
        self.ui.CameraPreview.clear()  # Clear any existing content
        self.ui.CameraPreview.setPixmap(QtGui.QPixmap())  # Clear the pixmap if needed
        self.ui.CameraPreview.setFixedSize(320, 240)  # Set the desired size

        # Create VideoCaptureWidget instance
        self.webcam = VideoCaptureWidget(self.ui.CameraPreview)
        self.webcam.setup_camera()
        self.webcam.set_preview_label(self.ui.CameraPreview)

        self.webcam.print_camera_info()

        # Populate the QComboBox with available cameras
        self.populate_camera_combobox()

        # Start the camera feed
        self.webcam.start_camera()

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
        self.ui.CreateGestureBtn.clicked.connect(self.launch_create_gesture_window)

        # stop camera feed
        if hasattr(self, 'webcam') and not self.ui.stackedWidget_2.currentWidget() == self.ui.page_5:
            self.webcam.stop_camera()

    def showPage7(self):
        self.ui.stackedWidget_2.setCurrentWidget(self.ui.page_7)
        self.highlight_button(self.ui.controllerBtn_5)
        self.ui.keyboardBtn_5.setStyleSheet("color: rgb(96, 100, 106);\n"
            "background-color: rgb(255, 255, 255);\n"
            "border:none;")
        self.ui.controlsSettingsBtn.setStyleSheet("background-color: rgb(99, 122, 155);\n"
                "color: rgb(255, 255, 255);\n"
                "border: none;")

        # stop camera feed
        if hasattr(self, 'webcam') and not self.ui.stackedWidget_2.currentWidget() == self.ui.page_5:
            self.webcam.stop_camera()

    def showPage8(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_8)
        self.highlight_button(self.ui.trackmaniaBtn)
        self.ui.launchGameBtn.clicked.connect(self.launch_trackmania)
        self.ui.launchGameBtn.clicked.connect(self.launch_camera_window)

        # stop camera feed
        if hasattr(self, 'webcam') and not self.ui.stackedWidget_2.currentWidget() == self.ui.page_5:
            self.webcam.stop_camera()

    def showPage9(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_9)

        # stop camera feed
        if hasattr(self, 'webcam') and not self.ui.stackedWidget_2.currentWidget() == self.ui.page_5:
            self.webcam.stop_camera()

    def showPage10(self):
        self.ui.stackedWidget_2.setCurrentWidget(self.ui.page_10)

        # stop camera feed
        if hasattr(self, 'webcam') and not self.ui.stackedWidget_2.currentWidget() == self.ui.page_5:
            self.webcam.stop_camera()

    def showPage11(self):
        self.ui.stackedWidget_2.setCurrentWidget(self.ui.page_11)

        # stop camera feed
        if hasattr(self, 'webcam') and not self.ui.stackedWidget_2.currentWidget() == self.ui.page_5:
            self.webcam.stop_camera()
    def launch_game_window(self):
        game_window = GameWindow(self.main_win, self.ui)
        game_window.show()

    def launch_camera_window(self):
        camera_window = CameraWindow(self.main_win, self.ui)
        if hasattr(self, 'webcam'):
            self.webcam.start_camera()
        camera_window.show()

    def launch_trackmania(self):
        # Specify the path to the Trackmania executable
        trackmania_path = r"C:\Program Files (x86)\Steam\steamapps\common\Trackmania\Trackmania.exe"

        # Specify the desired screen resolution
        resolution = "1600x900"  # Change this to your desired resolution

        # Use subprocess to launch Trackmania with the specified resolution
        subprocess.Popen([trackmania_path, f"-screen-width {resolution.split('x')[0]}", f"-screen-height {resolution.split('x')[1]}"])

    def launch_create_gesture_window(self):
        create_gesture_window = CreateGestureWindow(self.main_win, self.ui)

        # Connect the destroyed signal to stop the camera when the window is closed
        create_gesture_window.destroyed.connect(self.stop_camera_on_gesture_window_close)

        create_gesture_window.show()
        if hasattr(self, 'webcam'):
            self.webcam.start_camera()

    def stop_camera_on_gesture_window_close(self):
        # Stop the camera feed when the CreateGestureWindow is closed
        if hasattr(self, 'webcam'):
            self.webcam.stop_camera()

    def populate_camera_combobox(self):
        # Get the list of available cameras
        available_cameras = self.get_available_cameras()

        # Set the items in the QComboBox
        self.ui.CameraSelection.addItems(available_cameras)

        # Set the default selection to the first camera in the list
        if available_cameras:
            self.ui.CameraSelection.setCurrentIndex(0)

    def get_available_cameras(self):
        cameras = []
        for i in range(10):  # Assuming there are at most 10 cameras, adjust as needed
            camera = cv2.VideoCapture(i)
            if not camera.isOpened():
                break

            # Query additional camera properties
            backend_name = camera.getBackendName()
            backend_index = int(camera.get(cv2.CAP_PROP_BACKEND))
            width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Format the camera information
            camera_info = f"Camera {i} - {backend_name}, {backend_index}"

            camera.release()
            cameras.append(camera_info)

        return cameras


if __name__ == '__main__':
    # Enable High DPI display with PyQt5
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    main_win = MainWindow()

    main_win.show()
    sys.exit(app.exec_())