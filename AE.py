from comet_ml import Experiment
import torch
import torch.optim as optim
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.transforms import Compose
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import os

### SETTINGS
train_on_syntetic_data = False
log_on_comet = True
batch_size = 16
debug = False
###

class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, latent_dim):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_dim, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(h_dim, latent_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.conv_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose1d(in_dim, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(h_dim, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


class SPECDiscriminator(nn.Module):
    def __init__(self, n_samples=253074, sample_rate=16000):
        super(SPECDiscriminator, self).__init__()
        self.spectrogram_transform = Compose([
            MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=128, n_mels=128),
            AmplitudeToDB()
        ])

        def discriminator_block(in_filters, out_filters, kernel_size=3, stride=2, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size, stride, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, 3, 2, False),
            *discriminator_block(16, 32, 3, 2),
            *discriminator_block(32, 64, 3, 2),
            *discriminator_block(64, 128, 3, 2),
            *discriminator_block(128, 128, 3, 2),
            *discriminator_block(128, 128, 3, 2)
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(2816, 1),
            nn.Sigmoid()
        )

    def forward(self, audio_waveform):
        spectrogram = self.spectrogram_transform(audio_waveform)
        print(spectrogram.shape)
        out = self.model(spectrogram)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class Trainer:
    def __init__(self, autoencoder, spec_discriminator, spec_d_optimizer, ae_optimizer, criterion, adv_criterion, experiment, data_loader):
        self.autoencoder = autoencoder
        self.spec_discriminator = spec_discriminator
        self.ae_optimizer = ae_optimizer
        self.spec_d_optimizer = spec_d_optimizer
        self.criterion = criterion
        self.adv_criterion = adv_criterion
        self.experiment = experiment
        self.data_loader = data_loader
        self.mel_transform = Compose([
            MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=128, n_mels=128),
            AmplitudeToDB()
        ])


    def train(self, epochs=10):
        for epoch in range(epochs):
            for batch_idx, inputs in enumerate(self.data_loader):
                batch_size = inputs.size(0)

                # Train Discriminator
                self.spec_d_optimizer.zero_grad()

                # Forward pass for real inputs
                real_validity = self.spec_discriminator(inputs)
                real_labels = torch.ones(batch_size, 1)
                spec_d_real_loss = self.adv_criterion(real_validity, real_labels)

                # Forward pass for fake inputs
                outputs = self.autoencoder(inputs)
                fake_validity = self.spec_discriminator(outputs.detach())
                fake_labels = torch.zeros(batch_size, 1)
                spec_d_fake_loss = self.adv_criterion(fake_validity, fake_labels)

                # Compute Discriminator loss
                spec_d_loss = (spec_d_real_loss + spec_d_fake_loss) / 2
                spec_d_loss.backward()
                self.spec_d_optimizer.step()

                # Train Autoencoder
                self.ae_optimizer.zero_grad()
                outputs = self.autoencoder(inputs)
                ae_loss = self.criterion(outputs, inputs)
                
                # Generator loss
                fake_validity = self.spec_discriminator(outputs)
                generation_spec_loss = self.adv_criterion(fake_validity, real_labels)
                ae_g_loss = ae_loss + generation_spec_loss
                ae_g_loss.backward()
                self.ae_optimizer.step()

                # Logging every 5 batches
                if batch_idx % 5 == 0:
                    input_audio = inputs.cpu().detach().numpy().astype(np.float32)
                    output_audio = outputs.cpu().detach().numpy().astype(np.float32)
                    self.experiment.log_audio(audio_data=input_audio[0][0], sample_rate=16000, file_name=f'train_epoch_{epoch}_{batch_idx}.wav')
                    self.experiment.log_audio(audio_data=output_audio[0][0], sample_rate=16000, file_name=f'train_epoch_{epoch}_{batch_idx}_recons.wav')
                    
                    input_spectrogram = self.mel_transform(torch.tensor(input_audio[0][0])).cpu().detach().numpy()
                    output_spectrogram = self.mel_transform(torch.tensor(output_audio[0][0])).cpu().detach().numpy()
                    self.experiment.log_image(image_data=input_spectrogram, name=f'train_epoch_{epoch}_{batch_idx}_spec.png')
                    self.experiment.log_image(image_data=output_spectrogram, name=f'train_epoch_{epoch}_{batch_idx}_recons_spec.png')

                # Logging losses to Comet
                self.experiment.log_metric("spec_d_loss", spec_d_loss.item(), step=epoch * len(self.data_loader) + batch_idx)
                self.experiment.log_metric("ae_loss", ae_loss.item(), step=epoch * len(self.data_loader) + batch_idx)
                self.experiment.log_metric("generation_spec_loss", generation_spec_loss.item(), step=epoch * len(self.data_loader) + batch_idx)
                self.experiment.log_metric("ae_g_loss", ae_g_loss.item(), step=epoch * len(self.data_loader) + batch_idx)

                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(self.data_loader)}], SPEC_D Loss: {spec_d_loss.item()}, AE Loss: {ae_loss.item()}, SPEC_G Loss: {generation_spec_loss.item()}")

# Setup code to initialize models, optimizers, and other components
def setup_training(train_on_synthetic_data):
    in_dim = 1
    h_dim = 64
    latent_dim = 512
    sample_rate = 16000

    encoder = Encoder(in_dim, h_dim, latent_dim)
    decoder = Decoder(latent_dim, h_dim)
    autoencoder = nn.Sequential(encoder, decoder)

    spec_discriminator = SPECDiscriminator()
    
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.0003, betas=(0.5, 0.999))
    spec_d_optimizer = optim.Adam(spec_discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))

    criterion = nn.MSELoss()
    adv_criterion = nn.BCELoss()

    if log_on_comet:
        experiment = Experiment(api_key="usL9bE10QIu6K7Dokovu2MU3s",
                            project_name="emotional-speech")
    else:
       experiment = Experiment(api_key="NON_EXISTING_API_KEY",
                            project_name="WASD")

    
    if train_on_syntetic_data == False:
        data_path = "/content/drive/MyDrive/EmotionTransfer/Data/RAVDES"
        audio_dataset = RAVDESSDataset(data_path)
    else:
        audio_dataset = RandomAudioDatase(50)

    data_loader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True)
    
    trainer = Trainer(autoencoder, spec_discriminator, spec_d_optimizer, ae_optimizer, criterion, adv_criterion, experiment, data_loader)
    return trainer


class RAVDESSDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.filenames = []
        self.transform = transform
        self.target_length = 84358

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

class RandomAudioDatase(Dataset):
    def __init__(self, num_samples, target_lenght = 84358):
        self.num_samples = num_samples
        self.target_length = target_lenght

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        waveform = torch.rand(1,self.target_length)
        return waveform



# Initialize and start training
trainer = setup_training(train_on_syntetic_data)
trainer.train(epochs=10)
experiment.end()

