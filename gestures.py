# Final Year Project
# FYP-23-S4-30 : Gaming with Bare Hands
# A Machine Learning Approach to Vehicular Video Game Hand Gesture Controls

import mediapipe as mp
import cv2 as cv
import math
import input
import pyautogui
import json
import os
import numpy as np
import torch
from enum import Enum
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(nn.Linear(63, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20,11))

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class Labels(Enum):
    CLICK = 0
    CURSOR = 1
    FIST = 2
    FLAT = 3
    GUN = 4
    INWARD = 5
    OPENAWAY = 6
    OPENFACING = 7
    OUTWARD = 8
    THUMBSDOWN = 9
    THUMBSUP = 10

    def img2InputTensor(image, hands):
        right_hand_coordList = []
        left_hand_coordList = []

        # To improve performance, flag the image as unwriteable to reduce overhead
        image.flags.writeable = False

        # Convert from openCV BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # MediaPipe process the image
        results = hands.process(image)

        # Convert from RGB back to BGR for OpenCV
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks is None:
            return None, None, results

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Check if the hand is right or left and assign the landmarks to the respective list
            hand_type = handedness.classification[0].label
            coordList = right_hand_coordList if hand_type == 'Right' else left_hand_coordList

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                z = landmark.z

                coordList.extend([x, y, z])

        right_hand_tensor = torch.tensor(right_hand_coordList) if len(right_hand_coordList) == 63 else None
        left_hand_tensor = torch.tensor(left_hand_coordList) if len(left_hand_coordList) == 63 else None

        return right_hand_tensor, left_hand_tensor, results

class GestureRecognizer:
    def __init__(self):

        #Load the model
        self.model = NeuralNetwork()
        device = torch.device('cpu')
        self.model.to(device)

        # Get the directory of the script file
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the model file
        self.model_path = os.path.join(project_dir, 'model.pth')

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to inference mode

        # Set up Mediapipe hands object
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        # Set up Mediapipe draw styles and utilities
        self.mp_draw_styles = mp.solutions.drawing_styles
        self.mp_draw_utils = mp.solutions.drawing_utils

        # Set OpenCV font
        self.font = cv.FONT_HERSHEY_SIMPLEX

        # Initialize previous index finger tip coordinates
        self.prev_index_finger_tip_coords = None

    def RecognizeGestures(self, image):
        # Get opencv capture results
        # Returns a tuple success = bool representing the result of the read, and image is the capture
        #success, image = webcam_capture.read()
        #if not success:
            #print("Ignoring empty camera frame.")
            # Note: If loading a video, must use 'break' instead of 'continue'.
            #continue

        imgTensorRH, imgTensorLH, results = img2InputTensor(image, self.hands)

        #subtract the x values from 1 to mirror them
        for i in range(0, len(imgTensorLH), 3):
            imgTensorLH[i] = 1.0 - imgTensorLH[i]

        # Get Image Dimensions
        imageHeight, imageWidth, _ = image.shape

        # Re-flag the image as writeable
        image.flags.writeable = True

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #
        # Use the model to recognize the gesture on the right hand

        #pass the data in
        with torch.no_grad():  # We don't need to calculate gradients for inference
            if imgTensor is not None:
                #imgTensor.to(torch.device("cpu")) unnecessary
                predictions = self.model(imgTensor)

                predicted_index = predictions.argmax().item() #get the argmax index

                predicted_label = Labels(predicted_index) #convert to the Labels enum

                if predicted_label == Labels.CURSOR:
                    if index_finger_tip_coords:
                        index_finger_x, index_finger_y = index_finger_tip_coords[0]

                        # Scale the coordinates to match the screen size
                        scaled_x = int(index_finger_x * screen_width / imageWidth)

                        inverted_x = screen_width - scaled_x

                        scaled_y = int(index_finger_y * screen_height / imageHeight)

                        # Move the mouse cursor to the scaled position
                        pyautogui.moveTo(inverted_x, scaled_y)

                elif predicted_label == Labels.CLICK:
                        pyautogui.click()

                elif predicted_label == Labels.FIST:
                    input.release_key('s')
                    input.release_key('a')
                    input.release_key('d')
                    input.press_key('w')
                    cv.putText(image, "No turn.", (50, 50), self.font, 0.8, (0, 255, 0), 2, cv.LINE_AA)

                elif predicted_label == Labels.FLAT:

                elif predicted_label == Labels.GUN:

                elif predicted_label == Labels.INWARD:
                    print("Turn left.")
                    input.release_key('s')
                    input.release_key('d')
                    input.press_key('a')
                    cv.putText(image, "Turn left", (50, 50), self.font, 0.8, (0, 255, 0), 2, cv.LINE_AA)

                elif predicted_label == Labels.OPENAWAY:

                elif predicted_label == Labels.OPENFACING:
                    print("Reversing.")
                    input.release_key('a')
                    input.release_key('d')
                    input.release_key('w')
                    input.press_key('s')
                    cv.putText(image, "Reversing.", (50, 50), self.font, 1.0, (0, 255, 0), 2, cv.LINE_AA)

                elif predicted_label == Labels.OUTWARD:
                    print("Turn right.")
                    input.release_key('s')
                    input.release_key('a')
                    input.press_key('d')
                    cv.putText(image, "Turn right", (50, 50), self.font, 0.8, (0, 255, 0), 2, cv.LINE_AA)

                elif predicted_label == Labels.THUMBSDOWN:
                    print("Reversing Left.")
                    input.release_key('d')
                    input.release_key('w')
                    input.press_key('s')
                    input.press_key('a')
                    cv.putText(image, "Reversing Left.", (50, 50), self.font, 1.0, (0, 255, 0), 2, cv.LINE_AA)

                elif predicted_label == Labels.THUMBSUP:
                    print("Reversing Right.")
                    input.release_key('a')
                    input.release_key('w')
                    input.press_key('s')
                    input.press_key('d')
                    cv.putText(image, "Reversing Right.", (50, 50), self.font, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        #
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        cv.flip(image, 1)
        return image
        # Possible rate-limiting here?
        # Press Q to Quit
        #if cv.waitKey(5) & 0xFF == ord('q'):
            #break

    #Released webcam
    #webcam_capture.release()
    def __del__(self):
        """
        Destructor for cleaning up resources.
        """
        self.hands.close()
