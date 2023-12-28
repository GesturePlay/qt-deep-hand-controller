# Final Year Project
# FYP-23-S4-30 : Gaming with Bare Hands
# A Machine Learning Approach to Vehicular Video Game Hand Gesture Controls

import mediapipe as mp
import cv2 as cv
import math
import input
import pyautogui
import json

class GestureRecognizer:
    def __init__(self):
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

    def get_landmarks(self, image):
        thumb_tip_coords = []
        index_finger_tip_coords = []
        wrist_coords = []

        with self.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5) as hands:
            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw_utils.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                                      self.mp_draw_styles.get_default_hand_landmarks_style(),
                                                      self.mp_draw_styles.get_default_hand_connections_style())

                    for point in [self.mp_hands.HandLandmark.THUMB_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                  self.mp_hands.HandLandmark.WRIST]:
                        normalized_landmark = hand_landmarks.landmark[point]
                        pixel_coordinates_landmark = self.mp_draw_utils._normalized_to_pixel_coordinates(
                            normalized_landmark.x, normalized_landmark.y, image.shape[1], image.shape[0])

                        try:
                            if point == self.mp_hands.HandLandmark.THUMB_TIP:
                                thumb_tip_coords.append(list(pixel_coordinates_landmark))
                            elif point == self.mp_hands.HandLandmark.INDEX_FINGER_TIP:
                                index_finger_tip_coords.append(list(pixel_coordinates_landmark))
                            elif point == self.mp_hands.HandLandmark.WRIST:
                                wrist_coords.append(list(pixel_coordinates_landmark))
                        except Exception as e:
                            print(f"Error processing landmark {point}: {e}")

        return thumb_tip_coords, index_finger_tip_coords, wrist_coords

    def save_landmarks_to_file(self, landmarks, gesture_name):
        print("Landmarks to save:", landmarks)

        if landmarks is None:
            print("Landmarks are None. Unable to save to file.")
            return

        filename = f"{gesture_name}_landmarks.json"

        # Ensure landmarks is a list before saving
        if isinstance(landmarks, list):
            with open(filename, 'w') as json_file:
                json.dump(landmarks, json_file)

            print(f"Landmarks saved to {filename}")
        else:
            print("Landmarks should be a list. Unable to save to file.")

    def RecognizeGestures(self, image):
        with self.mp_hands.Hands(model_complexity = 0, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hands:


            # Get opencv capture results
            # Returns a tuple success = bool representing the result of the read, and image is the capture
            #success, image = webcam_capture.read()
            #if not success:
                #print("Ignoring empty camera frame.")
                # Note: If loading a video, must use 'break' instead of 'continue'.
                #continue

            # To improve performance, flag the image as unwriteable to reduce overhead
            image.flags.writeable = False

            # Convert from openCV BGR to RGB
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            # Media-pipe Process the image
            results = hands.process(image)

            # Get Image Dimensions
            screen_width, screen_height = 1920, 1080
            imageHeight, imageWidth, _ = image.shape

            # Re-flag the image as writeable
            image.flags.writeable = True

            # Convert from RGB back to BGR for OpenCV
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            # Create Empty Wrist Coords
            wrist_coords=[]

            # Create Empty Thumb and Index Finger Tips Coordinates
            thumb_tip_coords = []
            index_finger_tip_coords = []

            all_fingers_closed = False
            cursor_mode = False
            one_hand = False
            two_hand = False

            # If hand landmarks exist
            if results.multi_hand_landmarks:
                # Loop through hand landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the landmarks
                    self.mp_draw_utils.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                                      self.mp_draw_styles.get_default_hand_landmarks_style(),
                                                      self.mp_draw_styles.get_default_hand_connections_style())

                    # Loop through the points in the landmarks
                    for point in [self.mp_hands.HandLandmark.THUMB_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                  self.mp_hands.HandLandmark.WRIST]:
                        # Get the Normalized Landmark
                        normalizedLandmark = hand_landmarks.landmark[point]
                        # Use the Normalized landmark to get pixel coords based on the image size
                        pixelCoordinatesLandmark = self.mp_draw_utils._normalized_to_pixel_coordinates(
                            normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)

                        try:
                            # Try to add the pixel coordinates to the corresponding coords list
                            if point == self.mp_hands.HandLandmark.THUMB_TIP:
                                thumb_tip_coords.append(list(pixelCoordinatesLandmark))
                                print("Thumb Tip Coordinates:", list(pixelCoordinatesLandmark))
                            elif point == self.mp_hands.HandLandmark.INDEX_FINGER_TIP:
                                index_finger_tip_coords.append(list(pixelCoordinatesLandmark))
                                print("Index Finger Tip Coordinates:", list(pixelCoordinatesLandmark))
                            elif point == self.mp_hands.HandLandmark.WRIST:
                                wrist_coords.append(list(pixelCoordinatesLandmark))
                                print("Wrist Coordinates:", list(pixelCoordinatesLandmark))
                        except Exception as e:
                            print(f"Error processing landmark {point}: {e}")


                # Check if only one hand
                if len(results.multi_hand_landmarks) == 1:  # Check that only one hand is detected
                    one_hand = True

                # Check if there are two hands
                if len(results.multi_hand_landmarks) == 2: # Check that have 2 hands
                    two_hand = True

                # Check if all fingers are closed (thumb is above the rest of the fingers)
                if thumb_tip_coords and index_finger_tip_coords and all(
                        thumb_tip_coords[0][1] < finger_tip[1] for finger_tip in index_finger_tip_coords):
                    all_fingers_closed = True


                # Check if the hand is in an "L sign" position for cursor mode
                if thumb_tip_coords and index_finger_tip_coords:
                    thumb_tip_y = thumb_tip_coords[0][1]
                    index_finger_tip_y = index_finger_tip_coords[0][1]

                    # Check if the thumb tip is below all other fingers and the index tip is above all other fingers
                    if thumb_tip_y > max(finger_tip[1] for finger_tip in index_finger_tip_coords) and \
                       index_finger_tip_y < min(finger_tip[1] for finger_tip in thumb_tip_coords):
                        cursor_mode = True

                # Check if the thumb tip is close to the other fingertips (considered as a click)
                if thumb_tip_coords and index_finger_tip_coords:
                    thumb_tip_x, thumb_tip_y = thumb_tip_coords[0]
                    index_finger_tip_x, index_finger_tip_y = index_finger_tip_coords[0]

                    # Set a threshold for the distance (adjust as needed)
                    click_distance_threshold = 20

                    # Calculate the Euclidean distance between the thumb tip and index finger tip
                    distance = math.sqrt(
                        (thumb_tip_x - index_finger_tip_x) ** 2 + (thumb_tip_y - index_finger_tip_y) ** 2)

                    # Check if the distance is below the threshold
                    if distance < click_distance_threshold:
                        print("Thumb closed. Registering click.")
                        pyautogui.click()


            # If the "L sign" is detected, trigger the corresponding action
            while cursor_mode:
                #print("L sign detected. Updating mouse input.")

                if index_finger_tip_coords:
                    index_finger_x, index_finger_y = index_finger_tip_coords[0]

                    # Scale the coordinates to match the screen size
                    scaled_x = int(index_finger_x * screen_width / imageWidth)

                    inverted_x = screen_width - scaled_x

                    scaled_y = int(index_finger_y * screen_height / imageHeight)

                    # Move the mouse cursor to the scaled position
                    pyautogui.moveTo(inverted_x, scaled_y)

                    # Check if the thumb is closed (considered as a click)
                    if thumb_tip_coords:
                        thumb_tip_x, thumb_tip_y = thumb_tip_coords[0]

                        # Calculate the Euclidean distance between the thumb tip and index finger tip
                        distance = math.sqrt((thumb_tip_x - index_finger_x) ** 2 + (thumb_tip_y - index_finger_y) ** 2)

                        # Check if the distance is below the threshold
                        if distance < click_distance_threshold:
                            print("Thumb closed. Registering click.")
                            pyautogui.click()

                cursor_mode = False



            # If all fingers are closed and one wrist is not visible, we brake
            if all_fingers_closed and one_hand:
                print("Fist closed. Braking.")
                input.release_key('a')
                input.release_key('d')
                input.release_key('w')
                input.press_key('s')
                cv.putText(image, "Braking.", (50, 50), self.font, 1.0, (0, 255, 0), 2, cv.LINE_AA)

            # If we have two wrists on screen
            elif two_hand:

                # Alias to keep the code clean
                xy = wrist_coords

                # Get the means of the x and y coords for both wrists
                xmean, ymean = (xy[0][0] + xy[1][0]) / 2, (xy[0][1] + xy[1][1]) / 2

                try:
                    # Calculate the Slope
                    slope = (xy[1][1] - xy[0][1]) / (xy[1][0] - xy[0][0])
                except:
                    return image

                # Quadratic Function Coefficients
                a = 1 + slope ** 2
                b = -2 * xmean - 2 * xy[0][0] * (slope ** 2) + 2 * slope * xy[0][1] - 2 * slope * ymean
                c = xmean ** 2 + (slope ** 2) * (xy[0][0] ** 2) + xy[0][1] ** 2 + ymean ** 2 - 2 * xy[0][1] * ymean - 2 * xy[0][1] * xy[0][0] * slope + 2 * slope * ymean * xy[0][0] - 22500

                # Get X Intercepts using Quadratic Root
                xroot1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
                xroot2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)

                # Y Intercepts from X intercepts
                yroot1 = slope * (xroot1 - xy[0][0]) + xy[0][1]
                yroot2 = slope * (xroot2 - xy[0][0]) + xy[0][1]

                perp_xroot1 = 0
                perp_xroot2 = 0
                perp_yroot1 = 0
                perp_yroot2 = 0

                # If we are turning and wrists are not level
                if slope != 0:

                    #Quadratic coefficients for perpendicular
                    ap = 1 + ((-1 / slope) ** 2)
                    bp = -2 * xmean - 2 * xmean * ((-1 / slope) ** 2) + 2 * (-1 / slope) * ymean - 2 * (-1 / slope) * ymean
                    cp = xmean ** 2 + ((-1 / slope) ** 2) * (xmean ** 2) + ymean ** 2 + ymean ** 2 - 2 * ymean * ymean - 2 * ymean * xmean * (-1 / slope) + 2 * (-1 / slope) * ymean * xmean - 22500

                    try:
                        #X Intercepts for the Perpendicular Quadratic Equation
                        perp_xroot1 = (-bp + (bp ** 2 - 4 * ap * cp) ** 0.5) / (2 * ap)
                        perp_xroot2 = (-bp - (bp ** 2 - 4 * ap * cp) ** 0.5) / (2 * ap)

                        #Y Intercepts using the X Intercepts
                        perp_yroot1 = (-1 / slope) * (perp_xroot1 - xmean) + ymean
                        perp_yroot2 = (-1 / slope) * (perp_xroot2 - xmean) + ymean

                    except:
                        return image

                # Draw a circle with OpenCV
                cv.circle(img=image, center=(int(xmean), int(ymean)), radius=150, color=(195, 255, 62), thickness=15)

                # Get the distance between the wrists (not used?)
                #wrist_distance = (int(math.sqrt((xy[0][0] - xy[1][0]) ** 2 * (xy[0][1] - xy[1][1]) ** 2)) - 150) // 2

                # Draw lines with OpenCV
                cv.line(image, (int(xroot1), int(yroot1)), (int(xroot2), int(yroot2)), (195, 255, 62), 20)

                # Turn Left if...
                # If the right wrist is higher than the left and the difference is greater than 65

                if xy[0][0] > xy[1][0] and xy[0][1] > xy[1][1] and xy[0][1] - xy[1][1] > 65:
                    print("Turn left.")
                    input.release_key('s')
                    input.release_key('d')
                    input.press_key('a')
                    cv.putText(image, "Turn left", (50, 50), self.font, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.line(image, (int(perp_xroot2), int(perp_yroot2)), (int(xmean), int(ymean)), (195, 255, 62), 20)

                # We have to check for the case where the wrists are in a different order in the wrist_coord/xy list
                elif xy[1][0] > xy[0][0] and xy[1][1]> xy[0][1] and xy[1][1] - xy[0][1] > 65:
                    print("Turn left.")
                    input.release_key('s')
                    input.release_key('d')
                    input.press_key('a')
                    cv.putText(image, "Turn left", (50, 50), self.font, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.line(image, (int(perp_xroot2), int(perp_yroot2)), (int(xmean), int(ymean)), (195, 255, 62), 20)

                # Turn Right if...
                # If the left wrist is higher than the right and the difference is greater than 65
                elif xy[0][0] > xy[1][0] and xy[1][1]> xy[0][1] and xy[1][1] - xy[0][1] > 65:
                    print("Turn right.")
                    input.release_key('s')
                    input.release_key('a')
                    input.press_key('d')
                    cv.putText(image, "Turn right", (50, 50), self.font, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.line(image, (int(perp_xroot1), int(perp_yroot1)), (int(xmean), int(ymean)), (195, 255, 62), 20)

                # We have to check for the case where the wrists are in a different order in the wrist_coord/xy list
                elif xy[1][0] > xy[0][0] and xy[0][1]> xy[1][1] and xy[0][1] - xy[1][1] > 65:
                    print("Turn right.")
                    input.release_key('s')
                    input.release_key('a')
                    input.press_key('d')
                    cv.putText(image, "Turn right", (50, 50), self.font, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.line(image, (int(perp_xroot1), int(perp_yroot1)), (int(xmean), int(ymean)), (195, 255, 62), 20)

                # Else we do not turn.
                else:
                    print("No turn.")
                    input.release_key('s')
                    input.release_key('a')
                    input.release_key('d')
                    input.press_key('w')
                    cv.putText(image, "No turn.", (50, 50), self.font, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    if perp_yroot2 is not None and perp_xroot2 is not None and perp_xroot1 is not None and perp_yroot1 is not None:
                        if perp_yroot2 > perp_yroot1:
                            cv.line(image, (int(perp_xroot2), int(perp_yroot2)), (int(xmean), int(ymean)), (195, 255, 62), 20)
                        else:
                            cv.line(image, (int(perp_xroot1), int(perp_yroot1)), (int(xmean), int(ymean)), (195, 255, 62), 20)

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
