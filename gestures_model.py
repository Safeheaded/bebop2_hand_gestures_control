import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torchmetrics
import datetime

df = pd.read_csv('hand_landmarks.csv')

df = df.drop(columns=['landmark_index'])
df = df.drop(columns=['z'])

label_encoder = LabelEncoder()
df['klasa'] = label_encoder.fit_transform(df['klasa'])

finger_mapping = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3, 'pinky': 4}
df['finger'] = df['finger'].map(finger_mapping)
mylist = [[i+1]*19 for i in range(len(df))]
df['group_id'] = np.resize(mylist,len(df))
# One-hot encode the 'finger' column
grouped = df.groupby('group_id')

classes = []

group_arrays = []
for group_id, group in grouped:
    # Get the first coordinate to use as the new origin
    origin = group[['x', 'y']].iloc[0].values
    # Transform the coordinates
    group[['x', 'y']] = group[['x', 'y']] - origin

    # Normalize the coordinates
    min_vals = group[['x', 'y']].min()
    max_vals = group[['x', 'y']].max()
    group[['x', 'y']] = (group[['x', 'y']] - min_vals) / (max_vals - min_vals)

    group_array = group[['finger', 'x', 'y']].values
    group_arrays.append(group_array)
    classes.append(group['klasa'].iloc[0])

group_arrays_train, group_arrays_test, classes_train, classes_test = train_test_split(
    group_arrays, classes, test_size=0.2, random_state=42
)


class HandLandmarksDataset(Dataset):
    def __init__(self, group_arrays, classes):
        self.group_arrays = group_arrays
        self.classes = classes

    def __len__(self):
        return len(self.group_arrays)

    def __getitem__(self, idx):
        x = torch.tensor(self.group_arrays[idx], dtype=torch.float32)
        y = torch.tensor(self.classes[idx], dtype=torch.long)
        return x, y

class HandLandmarksModel(pl.LightningModule):
    def __init__(self, input_dim, num_classes):
        super(HandLandmarksModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

class HandLandmarksDataModule(pl.LightningDataModule):
    def __init__(self, group_arrays_train, classes_train, group_arrays_test, classes_test, batch_size=32):
        super().__init__()
        self.group_arrays_train = group_arrays_train
        self.classes_train = classes_train
        self.group_arrays_test = group_arrays_test
        self.classes_test = classes_test
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = HandLandmarksDataset(self.group_arrays_train, self.classes_train)
        self.val_dataset = HandLandmarksDataset(self.group_arrays_test, self.classes_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# Assuming group_arrays and classes are already defined as in your previous code
input_dim = group_arrays[0].shape[0] * group_arrays[0].shape[1]
num_classes = len(set(classes))

model = HandLandmarksModel(input_dim=input_dim, num_classes=num_classes)
data_module = HandLandmarksDataModule(group_arrays_train, classes_train, group_arrays_test, classes_test)

trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, data_module)

current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Define the file name with the date
file_name = f"hand_landmarks_model_{current_date}.pth"

# Save the model
torch.save(model.state_dict(), file_name)