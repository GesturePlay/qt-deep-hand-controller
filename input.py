import ctypes
from labels import Labels

keys = {
    "w": 0x11,
    "a": 0x1E,
    "s": 0x1F,
    "d": 0x20,
    "g": 0x22,  # Example value for 'g'
    "i": 0x17,  # Example value for 'i'
    "o": 0x18,  # Example value for 'o'
    "p": 0x19,  # Example value for 'p'
    "u": 0x16,  # Example value for 'u'
    "t": 0x14,  # Example value for 't'
    "z": 0x2C   # Example value for 'z'
}

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

    # Example usage
    #serialized_mapping = keymapping.serialize_mapping()
    def serialize_mapping(self):
        # Convert enum labels to their names for serialization
        serializable_mapping = {label.name: key_data for label, key_data in self.label_key_mapping.items()}
        return json.dumps(serializable_mapping)

    # Deserialize Usage
    #deserialized_mapping = keymapping.deserialize_mapping(serialized_mapping)
    #print(deserialized_mapping)
    def deserialize_mapping(self, serialized_mapping):
        # Convert from JSON and map back to Labels
        deserialized_mapping = json.loads(serialized_mapping)
        return {Labels[label]: (key_data[0], keys[key_data[0]]) for label, key_data in deserialized_mapping.items()}


    # NOTE THIS FUNCTION IS RETARDED
    #Usage: Change mapping
    #keymapping.change_mapping(Labels.CLICK, "d")
    def change_mapping(self, label, new_key):
        if label in self.label_key_mapping:
            if new_key in keys:
                self.label_key_mapping[label] = (new_key, keys[new_key])
            else:
                print(f"Key {new_key} not found in keys.")
        else:
            print(f"Label {label} not found in mapping.")

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

PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def press_key(key):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, keys[key], 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_key(key):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, keys[key], 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
