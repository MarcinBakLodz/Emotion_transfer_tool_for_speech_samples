import os
import torchaudio
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, latent_dim):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # 6 strided convolutional layers with stride 2 and window size 4
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
            # the latents consist of one feature map and the discrete space is 512-dimensional
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
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(h_dim, 1, kernel_size=6, stride=2, padding=1))

    def forward(self, x):
        return self.inverse_conv_stack(x)

class SPECDiscriminator(nn.Module):
    def __init__(self, n_samples = 253074, sample_rate = 16000):
        super(SPECDiscriminator, self).__init__()
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft = int(0.025 * sample_rate), hop_lenght = int(0.010* sample_rate), power =2.0)
        
        def discriminator_block(in_filters, out_filters, kernel_size=3, stride = 2, bn =True):
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
            nn.Linear(12800, 1),
            nn.Sigmoid()
        )
        
    def forward(self, audio_waveform):
        spectrogram = self.spectrogram_transform(audio_waveform)
        out = self.model(spectrogram)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
        
class Trainer:
    def __init__(self, autoencoder, spec_discriminator, spec_d_optimizer, ae_optimizer, criterion, adv_criterion):
        self.autoencoder = autoencoder
        self.spec_discriminator = spec_discriminator
        self.ae_optimizer = ae_optimizer
        self.spec_d_optimizer = spec_d_optimizer
        self.criterion = criterion
        self.adv_criterion = adv_criterion

    def train(self, inputs, epochs=10):
        for epoch in range(epochs):
            self.spec_d_optimizer.zero_grad()

            #Train discriminator
            # Forward pass
            real_validity = self.spec_discriminator(inputs)
            real_labels = torch.ones(batch_size, 1).to(inputs.device)
            spec_d_real_loss = self.adv_criterion(real_validity, real_labels)
            
            outputs = self.model(inputs)

            # Adjust output size to match input size if necessary
            if outputs.shape[2] < inputs.shape[2]:
                # Padding
                padding_needed = inputs.shape[2] - outputs.shape[2]
                outputs = torch.nn.functional.pad(outputs, (0, padding_needed))
            elif outputs.shape[2] > inputs.shape[2]:
                # Trimming
                outputs = outputs[:, :, :inputs.shape[2]]

            fake_validity = self.spec_discriminator(outputs.detach())
            fake_labels = torch.ones(batch_size, 1).to(inputs.device)
            spec_d_fake_loss = self.adv_criterion(fake_validity, fake_labels)
            

            # Compute loss
            spec_d_loss = (spec_d_real_loss + spec_d_fake_loss) / 2
            spec_d_loss.backward()
            self.spec_d_optimizer.step()
            loss = self.criterion(outputs, inputs)
            
            #Train AE
            self.ae_optimizer.zero_grad()
            
            outputs = self.model(inputs)

            # Adjust output size to match input size if necessary
            if outputs.shape[2] < inputs.shape[2]:
                # Padding
                padding_needed = inputs.shape[2] - outputs.shape[2]
                outputs = torch.nn.functional.pad(outputs, (0, padding_needed))
            elif outputs.shape[2] > inputs.shape[2]:
                # Trimming
                outputs = outputs[:, :, :inputs.shape[2]]
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            ae_loss = self.criterion(outputs, inputs)
            fake_validity = self.spec_discriminator(outputs.detach())
            generation_spec_loss = self.adv_critetion(fake_validity,real_labels)
            ae_g_loss = ae_loss + generation_spec_loss
            ae_g_loss.backward()
            self.ae_optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}], SPEC_D Loss: {spec_d_loss.item()}, AE Loss: {ae_loss.item()}, SPEC_G Loss: {generation_spec_loss.item()}")

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
spec_discriminator = SPECDiscriminator()

criterion = torch.nn.MSELoss()
adv_criterion = torch.nn.BCELoss()

ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
spec_d_optimizer = optim.Adam(spec_discriminator.parameters(), lr=0.001)

trainer = Trainer(autoencoder, spec_discriminator, ae_optimizer, spec_d_optimizer, criterion, adv_criterion)

for epoch in range(10):
    for inputs in data_loader:
        # You might need to adjust the input dimensions or preprocessing based on your network and data
        # inputs = inputs.view(inputs.size(0), 1, -1)  # Ensure input is in the correct shape
        trainer.train(inputs)