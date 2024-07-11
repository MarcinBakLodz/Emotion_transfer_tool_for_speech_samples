import os
import torchaudio
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

class Encoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, latent_dim):
        super().__init__()
        self.conv_stack = torch.nn.Sequential(
            # 6 strided convolutional layers with stride 2 and window size 4
            torch.nn.Conv1d(in_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            # the latents consist of one feature map and the discrete space is 512-dimensional
            torch.nn.Conv1d(h_dim, latent_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.conv_stack(x)


class Decoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()

        self.inverse_conv_stack = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(h_dim, 1, kernel_size=6, stride=2, padding=1))

    def forward(self, x):
        return self.inverse_conv_stack(x)


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, inputs, epochs=10):
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Adjust output size to match input size if necessary
            if outputs.shape[2] < inputs.shape[2]:
                # Padding
                padding_needed = inputs.shape[2] - outputs.shape[2]
                outputs = torch.nn.functional.pad(outputs, (0, padding_needed))
            elif outputs.shape[2] > inputs.shape[2]:
                # Trimming
                outputs = outputs[:, :, :inputs.shape[2]]

            # Compute loss
            loss = self.criterion(outputs, inputs)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

class AutoEncoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, h_dim, latent_dim)
        self.decoder = Decoder(latent_dim, h_dim)

    def forward(self, x):
        print("raw: ", x.shape)
        x = self.encoder(x)
        print("enc: ", x.shape)
        x = self.decoder(x)
        print("dec: ", x.shape)
        return x



class RAVDESSDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.filenames = []
        self.transform = transform
        self.target_length = 253074
        
        for actor in os.listdir(directory):
            actor_path = os.path.join(directory, actor)
            if os.path.isdir(actor_path):
                for filename in os.listdir(actor_path):
                    self.filenames.append(os.path.join(actor_path, filename))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_path = self.filenames[idx]
        waveform, sample_rate = torchaudio.load(audio_path, format="wav")
        # Crop or pad the waveform tensor to the target length
        if waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]  # Truncate the waveform
        elif waveform.shape[1] < self.target_length:
            padding_needed = self.target_length - waveform.shape[1]
            waveform = nn.functional.pad(waveform, (0, padding_needed))  # Pad with zeros on the last dimension
        return waveform

data_path = "./datasets/RAVDESS"

audio_dataset = RAVDESSDataset(data_path)

batch_size = 16
data_loader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True)


autoencoder = AutoEncoder(in_dim=1, h_dim=64, latent_dim=512)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
trainer = Trainer(autoencoder, criterion, optimizer)

for epoch in range(10):
    for inputs in data_loader:
        # You might need to adjust the input dimensions or preprocessing based on your network and data
        # inputs = inputs.view(inputs.size(0), 1, -1)  # Ensure input is in the correct shape
        trainer.train(inputs)
