import pyautogui
import json
from labels import Labels

class KeyMap:

    def __init__(self):
        self.label_key_mapping = {
            Labels.CLICK: "w",
            Labels.CURSOR: "a",
            Labels.FIST: "s",
            Labels.FLAT: "d",
            Labels.GUN: "g",
            Labels.INWARD: "i",
            Labels.OPENAWAY: "o",
            Labels.OPENFACING: "p",
            Labels.OUTWARD: "u",
            Labels.THUMBSUP: "t",
            Labels.THUMBSDOWN: "z",
        }

    # NOTE THIS FUNCTION IS RETARDED
    #Usage: Change mapping
    #keymapping.change_mapping(Labels.CLICK, "d")
    def change_mapping(self, label, new_key):
        if label in self.label_key_mapping:
            # Check if the new key is already assigned to another gesture
            for existing_label, existing_key in self.label_key_mapping.items():
                if existing_key == new_key:
                    # Update the mapping for the gesture that currently uses the new key
                    self.label_key_mapping[existing_label] = None

                if existing_label == label:
                    self.label_key_mapping[existing_label] = new_key
                    break

            self.label_key_mapping[label] = new_key
            print("Mapping changed successfully.")
            print("Updated label_key_mapping:", self.label_key_mapping)

        else:
            print(f"Label {label} not found in mapping.")

    def serialize(self):
        # Convert self.label_key_mapping to a JSON string

        label_int_key_mapping = {k.value : v for k,v in self.label_key_mapping.items()} #convert labels to integers for storage
        return json.dumps(label_int_key_mapping)

    @staticmethod
    def deserialize(keymap_str):
        keymap = KeyMap()
        if keymap_str:  # Check if the string is not empty or None
            keymap.label_key_mapping = {Labels(int(k)): v for k, v in
                                        json.loads(keymap_str).items()}  # Convert ints to labels from storage
        return keymap

class InputSimulator:
    def __init__(self, keymap):
        self.keymap = keymap
        self.pressed_keys = []

    def simulate_input(self, keys):
        for x in self.pressed_keys:
            if x not in keys:
                release_key(x)
        for x in keys:
            print(x)
            press_key(x)
        self.pressed_keys = keys

#press_key('a')  # Press the 'a' key
def press_key(key):
    pyautogui.keyDown(key)
#release_key('a') # Release the 'a' key
def release_key(key):
    pyautogui.keyUp(key)

