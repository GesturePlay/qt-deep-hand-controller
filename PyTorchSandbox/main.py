import os
import torch
from torch import nn
import numpy as np
from enum import Enum
import mediapipe as mp
import cv2 as cv
from PIL import Image

from torch.utils.data import (DataLoader, Dataset)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def convert_pil_to_opencv(pil_image):
    # Convert PIL image to a numpy array
    numpy_image = np.array(pil_image)

    # Convert RGB to BGR (which is how OpenCV stores image colors)
    opencv_image = cv.cvtColor(numpy_image, cv.COLOR_RGB2BGR)

    return opencv_image

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
    THUMBSUP = 9
    THUMBSDOWN = 10

class CustomDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []

        with mp.solutions.hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

            data_directory = r"C:\Users\ENA\My Drive\school\Final Year Project\qt-deep-hand-controller\data"

            # Define a mapping from filename prefixes to Labels enum
            label_mapping = {
                "click": Labels.CLICK,
                "cursor": Labels.CURSOR,
                "fist": Labels.FIST,
                "flat": Labels.FLAT,
                "gun": Labels.GUN,
                "inward": Labels.INWARD,
                "openaway": Labels.OPENAWAY,
                "openfacing": Labels.OPENFACING,
                "outward": Labels.OUTWARD,
                "thumbsup": Labels.THUMBSUP,
                "thumbsdown": Labels.THUMBSDOWN,
            }

            file_number = 0
            total_files = sum(
                [len(files) for r, d, files in os.walk(data_directory) if any(file.endswith(".jpg") for file in files)])

            # Recursively walk through the directory tree
            for root, dirs, files in os.walk(data_directory):
                for filename in files:
                    if filename.endswith(".jpg"):
                        file_number += 1
                        print("Loading", filename, "- Image:", file_number, "/", total_files)

                        # Construct the full file path
                        image_path = os.path.join(root, filename)
                        image = Image.open(image_path)

                        # Recognize the label from the filename
                        for prefix, label_enum in label_mapping.items():
                            if filename.startswith(prefix):
                                # Process your image here
                                opencv_image = convert_pil_to_opencv(image)
                                imgTensor = CustomDataset.img2InputTensor(opencv_image, hands)
                                if imgTensor is not None:
                                    self.data.append(imgTensor)
                                    self.labels.append(label_enum.value)
                                break  # Found the correct label, no need to continue the inner loop

            self.data = torch.stack(self.data)
            self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.labels[idx]
        return item, label

    @staticmethod
    def img2InputTensor(image, hands):

        coordList = []

        # To improve performance, flag the image as unwriteable to reduce overhead
        image.flags.writeable = False

        # Convert from openCV BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Media-pipe Process the image
        results = hands.process(image)

        # Convert from RGB back to BGR for OpenCV
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks is None:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]


        for landmark in hand_landmarks.landmark:
            # Each 'landmark' here represents a point on the hand.
            # Store the x, y coordinates of each point.
            x = landmark.x
            y = landmark.y
            z = landmark.z

            # Append the coordinates to the single hand's list.
            coordList.append(x)
            coordList.append(y)
            coordList.append(z)

        if len(coordList) == 63:
            return torch.tensor(coordList)
        else:
            raise Exception("Did not detect exactly 63 coordinates")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.network = model

    def forward(self, x):
        logits = self.network(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

dataset = CustomDataset()

def TrainModel(architecture, batchSize, learningRate, numEpochs, optimizer_type='SGD'): #batchsize 50, 1e-3 for LR, numEpochs 800
    # Create data loaders.
    train_dataloader = DataLoader(dataset, batch_size=batchSize)
    test_dataloader = DataLoader(dataset, batch_size=batchSize)

    model = NeuralNetwork(architecture).to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()

    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    test(test_dataloader, model, loss_fn)
    for t in range(numEpochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
        print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")