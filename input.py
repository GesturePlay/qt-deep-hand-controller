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
            self.save_to_file()  # Save changes immediately when a mapping is changed
            print("Mapping changed successfully.")
        else:
            print(f"Label {label} not found in mapping.")

    def save_to_file(self):
        # Convert the label_key_mapping dictionary to JSON string
        keymap_json = json.dumps({label.value: key for label, key in self.label_key_mapping.items()})

        # Write the JSON string to the file
        try:
            with open("gesture_mappings.json", "w") as f:
                f.write(keymap_json)
            print("Changes saved to file.")
        except Exception as e:
            print(f"Error saving changes to file: {e}")



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
