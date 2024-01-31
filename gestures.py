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
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from labels import Labels

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(63, 100), nn.ELU(), nn.Linear(100, 11)) #1000adam-finished


    def forward(self, x):
        logits = self.network(x)
        return logits

    @staticmethod
    def img2InputTensor(image, hands):
        # Ensure image is valid
        if image is None:
            print("Image is None, cannot process.")
            return None, None, None

        right_hand_coordList = []
        left_hand_coordList = []

        image.flags.writeable = True

        # Convert from openCV BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # MediaPipe process the image
        results = hands.process(image)

        # Convert from RGB back to BGR for OpenCV
        #image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if results is None:
            print("MediaPipe Hands returned None.")
            return None, None, None

        if results.multi_hand_landmarks is None:
            print("Multi hand Landmarks results none.")
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

        #for some odd reason we have to swap the hands here perhaps because the image is not flipped yet
        right_hand_tensor, left_hand_tensor = left_hand_tensor, right_hand_tensor

        # To improve performance, flag the image as unwriteable to reduce overhead
        image.flags.writeable = False

        return right_hand_tensor, left_hand_tensor, results

class GestureRecognizer:
    def __init__(self):

        #Load the model
        self.model = NeuralNetwork()
        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.gesture_buffer = {"RH": [], "LH": []}

        # Get the directory of the script file
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the model file
        self.model_path = os.path.join(project_dir, '1000adam-finished.pth')                       #path to model

        try:
            self.model.load_state_dict(torch.load(self.model_path))
        except Exception as e:
            print(f"Error loading the model: {e}")

        self.model.eval()  # Set the model to inference mode


        # Set up Mediapipe hands object
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        # Set up Mediapipe draw styles and utilities
        self.mp_draw_styles = mp.solutions.drawing_styles
        self.mp_draw_utils = mp.solutions.drawing_utils

        # Set OpenCV font
        self.font = cv.FONT_HERSHEY_SIMPLEX

        self.label_confidence_map = {
            Labels.CLICK: 0.6,
            Labels.CURSOR:0.6,
            Labels.FIST: 0.6,
            Labels.FLAT: 0.5,
            Labels.GUN: 0.5,
            Labels.INWARD: 0.6,
            Labels.OPENAWAY: 0.7,
            Labels.OPENFACING: 0.7,
            Labels.OUTWARD: 0.6,
            Labels.THUMBSUP: 0.6,
            Labels.THUMBSDOWN: 0.6
        }

    def update_gesture_buffer(self, hand, new_gesture):
        if hand not in self.gesture_buffer:
            # Initialize buffer for this hand if it doesn't exist
            self.gesture_buffer[hand] = []

        # Append the new gesture
        self.gesture_buffer[hand].append(new_gesture)

        # Ensure only the last two gestures are kept
        while len(self.gesture_buffer[hand]) > 2:
            self.gesture_buffer[hand].pop(0)

    def is_consecutive_match(self, hand):
        """
        Check if the last two gestures are the same.
        """
        # Check if the buffer for the specified hand is initialized and contains two elements
        if hand in self.gesture_buffer and len(self.gesture_buffer[hand]) == 2:
            # Retrieve the last two gestures
            last_gesture, previous_gesture = self.gesture_buffer[hand]

            # Check if neither of the gestures is None and if they are equal
            if last_gesture is not None and previous_gesture is not None:
                return last_gesture == previous_gesture

        # Return False if the conditions are not met
        return False

    def recognize_gestures(self, image):

        if image is not None:
            imageHeight, imageWidth, _ = image.shape
            imgTensorRH, imgTensorLH, results = NeuralNetwork.img2InputTensor(image, self.hands)

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #
        # Use the model to recognize the gestures

        #pass the data in
        with torch.no_grad():  # We don't need to calculate gradients for inference

            predicted_label_RH, predicted_label_LH,  = None, None
            confirmed_label_LH, confirmed_label_RH = None, None

            if imgTensorLH is not None:
                #imgTensor.to(torch.device("cpu")) unnecessary
                #subtract the x values from 1 to mirror them

                for i in range(0, len(imgTensorLH), 3):
                    imgTensorLH[i] = 1.0 - imgTensorLH[i]

                predictions_LH = self.model(imgTensorLH)
                predicted_index_LH = predictions_LH.argmax().item() #get the argmax index
                predicted_label_LH = Labels(predicted_index_LH) #convert to the Labels enum

                # Ensure predictions are probabilities
                probabilities_LH = F.softmax(predictions_LH, dim=0)
                confidence_LH = probabilities_LH[predicted_index_LH].item()  # Assuming batch size is 1

                #if confidence_LH > self.label_confidence_map[predicted_label_LH]: #append gesture to buffer when confident
                self.update_gesture_buffer("LH", predicted_label_LH)
                #else:
                    #self.update_gesture_buffer("LH", None) #if not confident add none gesture to buffer

                if self.is_consecutive_match("LH"):
                    confirmed_label_LH = predicted_label_LH
                    print("LH: ", predicted_label_LH, " Confidence: ", confidence_LH)


            if imgTensorRH is not None:
                predictions_RH = self.model(imgTensorRH)
                predicted_index_RH = predictions_RH.argmax().item() #get the argmax index
                predicted_label_RH = Labels(predicted_index_RH) #convert to the Labels enum

                # Ensure predictions are probabilities
                probabilities_RH = F.softmax(predictions_RH, dim=0)
                confidence_RH = probabilities_RH[predicted_index_RH].item()  # Assuming batch size is 1

                #if confidence_RH > self.label_confidence_map[predicted_label_RH]: #append gesture to buffer when confident
                self.update_gesture_buffer("RH", predicted_label_RH)
                #else:
                    #self.update_gesture_buffer("RH", None) #if not confident add none gesture to buffer

                if self.is_consecutive_match("RH"):
                    confirmed_label_RH = predicted_label_RH
                    print("RH: ", predicted_label_RH, " Confidence: ", confidence_RH)

            if Labels.CURSOR in [predicted_label_LH, predicted_label_RH]:

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_type = handedness.classification[0].label
                    index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    screen_width, screen_height = pyautogui.size()

                    scaled_x = int(index_finger.x * screen_width)
                    inverted_x = screen_width - scaled_x
                    scaled_y = int(index_finger.y * screen_height)

                    #because hand recognition is backwards we have to check for right hand for lh labels and vice versa
                    if Labels.CURSOR == predicted_label_LH and hand_type == 'Right':
                        pyautogui.moveTo(inverted_x, scaled_y)
                    elif Labels.CURSOR == predicted_label_RH and hand_type == 'Left':
                        pyautogui.moveTo(inverted_x, scaled_y)

                #return no gestures
                return None, None

            if Labels.CLICK in [predicted_label_LH, predicted_label_RH]:
                pyautogui.click()
                return None, None

            return confirmed_label_LH, confirmed_label_RH

        #Model end
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #cv.putText(image, "Turn left", (50, 50), self.font, 0.8, (0, 255, 0), 2, cv.LINE_AA)

    def __del__(self):
        """
        Destructor for cleaning up resources.
        """
        self.hands.close()
