import sys

import labels
from gestures import GestureRecognizer
from PyQt5.QtCore import Qt, QSettings
from app import Ui_MainWindow
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QInputDialog, QComboBox, QVBoxLayout, QWidget, QFormLayout, QDialog, QGridLayout, QMessageBox

from PyQt5.QtCore import QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QStandardItemModel, QStandardItem
import subprocess
import cv2
import json
import os
import numpy as np
from labels import Labels
from input import InputSimulator
from input import KeyMap
from profile import UserProfile


userProfiles = UserProfile.deserialize_user_profiles()

if userProfiles == []:
    userProfiles.append(UserProfile()) #append a default user if there are no users

#need to set the active user here with a function when the user selects their profile at the main window
activeUserProfile = userProfiles[0] #need to store the current user in this global, defaults to default new user


class VideoCaptureWidget(QtWidgets.QWidget):
    def __init__(self, label, parent=None):
        super(VideoCaptureWidget, self).__init__(parent)
        self.input_simulator = InputSimulator(activeUserProfile.keymap)
        self.hands_recognizer = GestureRecognizer()
        self.video_size = QSize(320, 240)
        self.image_label = label
        self.setup_ui()
        self.setup_camera()
        self.active = False

        # Connect the update_frame method to the timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms


    def setup_ui(self):
        """Initialize widgets."""
        self.image_label.setFixedSize(self.video_size)

        # Set the layout
        #layout = QtWidgets.QVBoxLayout()
        #layout.addWidget(self.image_label)
        #self.setLayout(layout)

    def setup_camera(self, camera_index=0):
        """Set up the camera with the specified index."""
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

    def update_frame(self):
        """Capture frame from the webcam and update QLabel."""
        ret, frame = self.capture.read()
        if ret:
            #recognize the gestures
            lh_label, rh_label = self.hands_recognizer.recognize_gestures(frame)
            #convert to key
            rh_key = None if rh_label is None else activeUserProfile.keymap.label_key_mapping[rh_label]
            lh_key = None if lh_label is None else activeUserProfile.keymap.label_key_mapping[lh_label]
            #simulate input (press and release) for the keys
            y = [x for x in [rh_key, lh_key] if x is not None]
            self.input_simulator.simulate_input([x for x in [rh_key, lh_key] if x is not None])
            # Convert and Show the frame in the QLabel
            cv2.flip(frame, 1)
            frame = self.convert_cv_qt(frame)
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

class ImageSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super(ImageSelectionDialog, self).__init__(parent)
        self.setWindowTitle("Choose Gesture")
        self.setGeometry(800, 500, 200, 150)

        # Create a layout for the dialog
        layout = QVBoxLayout()

        # Create a combo box to display a group of images
        self.comboBox = QComboBox()
        self.populate_combo_box()
        layout.addWidget(self.comboBox)

        # Create a button to confirm the selection
        confirm_button = QPushButton("Select")
        confirm_button.clicked.connect(self.accept)
        layout.addWidget(confirm_button)

        self.setLayout(layout)

    def populate_combo_box(self):
        # Assuming you have a list of image paths
        image_paths = ["Images/Icons/click.png", "Images/Icons/cursor.png", "Images/Icons/Fist.png", "Images/Icons/Flat.png", "Images/Icons/Gun.png", "Images/Icons/Inward.png", "Images/Icons/FacingAway.png", "Images/Icons/FacingTowards.png", "Images/Icons/Outward.png", "Images/Icons/ThumbsUp.png", "Images/Icons/ThumbsDown.png"]

        for path in image_paths:
            # Use QStandardItem for each item in the combo box
            item = QStandardItem(path.split("/")[-1])  # Display only the filename
            item.setData(path, Qt.UserRole)  # Store the full path as user data
            self.comboBox.model().appendRow(item)

    def selected_image_path(self):
        # Get the selected item from the combo box
        index = self.comboBox.currentIndex()
        item = self.comboBox.model().item(index)

        if item is not None:
            # Print the index of the selected gesture
            print("Index of selected gesture:", index)

            # Return the path of the selected image
            return item.data(Qt.UserRole)
        return None

    def selected_image_label(self):
        # Get the selected item from the combo box
        index = self.comboBox.currentIndex()
        item = self.comboBox.model().item(index)

        if item is not None:
            # Return the path of the selected image
            return Labels(index)
        return None

class MainWindow:
    def __init__(self):
        self.settings = QSettings("YourCompany", "YourApp")
        self.main_win = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)
        self.keymap = KeyMap()
        self.input_simulator = InputSimulator(self.keymap)  # Pass keymap instance here

        #layout = QFormLayout(self.main_win.centralWidget())

        #self.webcam = VideoCaptureWidget(self.ui.webcamLabel)
        # Connect button clicks to show pages and highlight buttons
        self.setup_button(self.ui.Profile1, self.showPage2)
        self.setup_button(self.ui.gameSelectionBtn, self.showPage3)
        self.setup_button(self.ui.cameraSettingsBtn, self.showPage5)
        self.setup_button(self.ui.controlsSettingsBtn, self.showPage6)
        self.setup_button(self.ui.logoutBtn, self.showPage1)
        self.setup_button(self.ui.trackmaniaBtn, self.showPage8)
        self.setup_button(self.ui.controlsSettingsBtn_3, self.showPage2)
        self.setup_button(self.ui.logoutBtn_2, self.showPage1)

        # Set initial page and highlight the corresponding button
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_1)
        self.highlight_button(self.ui.logoutBtn)


        self.label_key_mapping = {
            self.ui.imgLabel1: "w",
            self.ui.imgLabel2: "a",
            self.ui.imgLabel3: "s",
            self.ui.imgLabel4: "d",
            self.ui.imgLabel5: "g",
            self.ui.imgLabel6: "i",
            self.ui.imgLabel7: "o",
            self.ui.imgLabel8: "p",
            self.ui.imgLabel9: "u"
        }

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
        for btn in [self.ui.Profile1, self.ui.gameSelectionBtn,
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

        for i in range(1, 10):
            img_label = getattr(self.ui, f"imgLabel{i}")
            # Load saved image path from settings
            saved_image_path = self.settings.value(f"image_path_{i}", "")
            if saved_image_path:
                img_label.setPixmap(QPixmap(saved_image_path))
            img_label.mousePressEvent = lambda event, idx=i: self.image_clicked(idx)

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

    def launch_game_window(self):
        game_window = GameWindow(self.main_win, self.ui)
        game_window.show()

    def launch_camera_window(self):
        camera_window = CameraWindow(self.main_win, self.ui)
        if hasattr(self, 'webcam'):
            self.webcam.start_camera()
        camera_window.show()

    def launch_trackmania(self):
        # Get the current working directory
        trackmania_path = r"C:\Program Files (x86)\Steam\steamapps\common\Trackmania\Trackmania.exe"

        # Specify the desired screen resolution
        resolution = "1600x900"  # Change this to your desired resolution

        subprocess.Popen([trackmania_path, f"-screen-width {resolution.split('x')[0]}", f"-screen-height {resolution.split('x')[1]}"])

    def populate_camera_combobox(self):
        # Get the list of available cameras
        available_cameras = self.get_available_cameras()

        # Set the items in the QComboBox
        self.ui.CameraSelection.addItems(available_cameras)

        # Connect the currentIndexChanged signal to the slot method
        self.ui.CameraSelection.currentIndexChanged.connect(self.camera_selection_changed)

        # Set the default selection to the first camera in the list
        if available_cameras:
            self.ui.CameraSelection.setCurrentIndex(0)

    def camera_selection_changed(self, index):
        # Stop the current camera feed
        if hasattr(self, 'webcam'):
            self.webcam.stop_camera()
            self.webcam.close()  # Close the existing camera instance

        # Create a new VideoCaptureWidget instance with the selected camera index
        self.webcam = VideoCaptureWidget(self.ui.CameraPreview)
        selected_camera_index = self.ui.CameraSelection.currentIndex()
        self.webcam.setup_camera(camera_index=selected_camera_index)
        self.webcam.set_preview_label(self.ui.CameraPreview)

        # Start the camera feed
        self.webcam.start_camera()

        # Print the selected camera index
        print("Selected camera index:", selected_camera_index)



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

    def image_clicked(self, idx):
        # Show the image selection dialog with self.main_win as the parent
        dialog = ImageSelectionDialog(self.main_win)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            # Get the selected image path from the dialog
            selected_image_path = dialog.selected_image_path()
            selected_image_label = dialog.selected_image_label()

            if selected_image_path is not None:
                # Load and set the new image in the corresponding label
                img_label = getattr(self.ui, f"imgLabel{idx}")
                img_label.setPixmap(QPixmap(selected_image_path))
                self.settings.setValue(f"image_path_{idx}", selected_image_path)

                # Get the corresponding key for the label
                selected_key = self.label_key_mapping.get(img_label)

                print(selected_image_path)
                print("Selected label:", selected_image_label)
                print("Selected key:", selected_key)

                activeUserProfile.keymap.change_mapping(selected_image_label, selected_key)

                # Loop through other imgLabels and print if their pixmap paths match
                for i in range(1, 10):
                    if i != idx:  # Skip the current label
                        other_img_label = getattr(self.ui, f"imgLabel{i}")
                        # Get the image path of the other label from the settings
                        other_image_path = self.settings.value(f"image_path_{i}", "")
                        if other_img_label.pixmap().toImage() == img_label.pixmap().toImage():
                            print(f"Image label {i} has the same image path")
                            other_img_label.setPixmap(QPixmap("Images/Icons/unassigned.png"))  # Set to unassigned.png
                            # Save the change to the setting
                            self.settings.setValue(f"image_path_{i}", "Images/Icons/unassigned.png")


if __name__ == '__main__':
    # Enable High DPI display with PyQt5
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    result = app.exec_()
    UserProfile.serialize_user_profiles(userProfiles)
    sys.exit(result)