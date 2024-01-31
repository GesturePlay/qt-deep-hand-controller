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
            self.label_key_mapping[label] = new_key
        else:
            print(f"Label {label} not found in mapping.")
    def serialize(self):
        # Convert self.label_key_mapping to a JSON string

        label_int_key_mapping = {k.value : v for k,v in self.label_key_mapping.items()} #convert labels to integers for storage
        return json.dumps(label_int_key_mapping)

    @staticmethod
    def deserialize(keymap_str):
        # Convert JSON string back to a dictionary
        keymap = KeyMap()
        keymap.label_key_mapping = {Labels(int(k)) : v for k,v in json.loads(keymap_str).items()} #convert ints to labels from storage
        return keymap

class InputSimulator:
    def __init__(self):
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
