from input import KeyMap
from labels import Labels
import json
import os

class UserProfile:
    def __init__(self, userName = "New User", keyMap = KeyMap(), gameList = []):
        self.username = userName #a string
        self.keymap = keyMap #contains a dictionary self.label_key_mapping with dictionary-keys of gesture Labels and values of keycodes
        self.gamelist = gameList #a list of executables

    #Example Usage: serialized_user = user.serialize()
    def serialize(self):
        # Convert the UserProfile object into a JSON string
        return json.dumps({
            'username': self.username,
            'keymap': self.keymap.serialize(),
            'gamelist': self.gamelist
        })

    @staticmethod
    def deserialize(user_profile_str):
        # Convert the JSON string back into a UserProfile object
        data = json.loads(user_profile_str)
        user_profile = UserProfile(data['username'])
        user_profile.keymap = KeyMap.deserialize(data['keymap'])
        user_profile.gamelist = data['gamelist']
        return user_profile

    @staticmethod
    def deserialize_user_profiles():
        user_profiles = []

        # Construct the full path to the config folder
        # Assuming the config folder is directly under the directory of the main script
        base_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_path, "config")

        # Iterate over all files in the config folder
        for filename in os.listdir(full_path):
            if filename.endswith(".json"):
                file_path = os.path.join(full_path, filename)

                # Read the content of the file and deserialize it
                with open(file_path, 'r') as file:
                    user_profile_str = file.read()
                    user_profile = UserProfile.deserialize(user_profile_str)
                    user_profiles.append(user_profile)

        return user_profiles

    @staticmethod
    def serialize_user_profiles(user_profiles):
        # Construct the full path to the config folder
        base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, "config")

        # Create the config directory if it doesn't exist
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        for profile in user_profiles:
            # Create a filename for each profile
            # Assuming username is unique and valid for filename
            file_name = f"{profile.username}.json"
            file_path = os.path.join(config_path, file_name)

            # Serialize the UserProfile object
            serialized_data = profile.serialize()

            # Write the serialized data to a file
            with open(file_path, 'w') as file:
                file.write(serialized_data)