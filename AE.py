from comet_ml import Experiment
import torch
import torch.optim as optim
import torchaudio
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchaudio.transforms import MelSpectrogram

class Encoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, latent_dim):
        super().__init__()
        self.conv_stack = torch.nn.Sequential(
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
    def __init__(self, model, criterion, optimizer, experiment):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.experiment = experiment
        self.mel_transform = MelSpectrogram(sample_rate=16000, n_mels=128)

    def train(self, data_loader, epochs=10):
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, inputs in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                
                if outputs.shape[2] < inputs.shape[2]:
                    padding_needed = inputs.shape[2] - outputs.shape[2]
                    outputs = torch.nn.functional.pad(outputs, (0, padding_needed))
                elif outputs.shape[2] > inputs.shape[2]:
                    outputs = outputs[:, :, :inputs.shape[2]]

                # Compute loss
                loss = self.criterion(outputs, inputs)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if batch_idx % 5 == 0:
                    input_audio = inputs.cpu().detach().numpy().astype(np.float32)
                    output_audio = outputs.cpu().detach().numpy().astype(np.float32)
                    self.experiment.log_audio(audio_data=input_audio[0][0], sample_rate=16000, file_name=f'train_epoch_{epoch}_{batch_idx}.wav')
                    self.experiment.log_audio(audio_data=output_audio[0][0], sample_rate=16000, file_name=f'train_epoch_{epoch}_{batch_idx}_recons.wav')
                    
                    input_spectrogram = self.mel_transform(torch.tensor(input_audio[0][0])).cpu().detach().numpy()
                    output_spectrogram = self.mel_transform(torch.tensor(output_audio[0][0])).cpu().detach().numpy()
                    self.experiment.log_image(image_data=input_spectrogram, name=f'train_epoch_{epoch}_{batch_idx}_spec.png')
                    self.experiment.log_image(image_data=output_spectrogram, name=f'train_epoch_{epoch}_{batch_idx}_recons_spec.png')

            avg_loss = epoch_loss / len(data_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss}")

class AutoEncoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, h_dim, latent_dim)
        self.decoder = Decoder(latent_dim, h_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class RAVDESSDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.filenames = []
        self.transform = transform
        
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
        return waveform

data_path = "./datasets/RAVDESS"
audio_dataset = RAVDESSDataset(data_path)

batch_size = 16
data_loader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True)

experiment = Experiment(api_key="YOUR_API_KEY",
                        project_name="YOUR_PROJECT_NAME")

autoencoder = AutoEncoder(in_dim=1, h_dim=64, latent_dim=512)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
trainer = Trainer(autoencoder, criterion, optimizer, experiment)

trainer.train(data_loader, epochs=10)
