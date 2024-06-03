import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
import torch
import torch.optim as optim
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from torchvision.transforms import Compose
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import os
from pytorch_lightning.callbacks import EarlyStopping

### SETTINGS
train_on_syntetic_data = False
log_on_comet = True
batch_size = 16
debug = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(torch.nn.functional.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class SoundStreamResidualUnit(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.dilation = dilation

        self.layers = torch.nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, dilation=dilation),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)

class SoundStreamEncoderBlock(torch.nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = torch.nn.Sequential(
            SoundStreamResidualUnit(in_channels=out_channels//2, out_channels=out_channels//2, dilation=1),
            torch.nn.ELU(),
            SoundStreamResidualUnit(in_channels=out_channels//2, out_channels=out_channels//2, dilation=3),
            torch.nn.ELU(),
            SoundStreamResidualUnit(in_channels=out_channels//2, out_channels=out_channels//2, dilation=9),
            torch.nn.ELU(),
            CausalConv1d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=2*stride, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)

class SoundStreamEncoder(torch.nn.Module):
    def __init__(self, C, D):
        super().__init__()

        self.layers = torch.nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),
            torch.nn.ELU(),
            SoundStreamEncoderBlock(out_channels=2*C, stride=2),
            torch.nn.ELU(),
            SoundStreamEncoderBlock(out_channels=4*C, stride=4),
            torch.nn.ELU(),
            SoundStreamEncoderBlock(out_channels=8*C, stride=5),
            torch.nn.ELU(),
            SoundStreamEncoderBlock(out_channels=16*C, stride=7),
            torch.nn.ELU(),
            CausalConv1d(in_channels=16*C, out_channels=D, kernel_size=3)
        )

    def forward(self, x):
        y = self.layers(x)
        return y

class CausalConvTranspose1d(torch.nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]

    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return torch.nn.functional.conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)[..., :-self.causal_padding]

class SoundStreamDecoderBlock(torch.nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = torch.nn.Sequential(
            CausalConvTranspose1d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=2*stride, stride=stride),
            torch.nn.ELU(),
            SoundStreamResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1),
            torch.nn.ELU(),
            SoundStreamResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3),
            torch.nn.ELU(),
            SoundStreamResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9),
        )

    def forward(self, x):
        return self.layers(x)

class SoundStreamDecoder(torch.nn.Module):
    def __init__(self, C, D):
        super().__init__()

        self.layers = torch.nn.Sequential(
            CausalConv1d(in_channels=D, out_channels=16*C, kernel_size=7),
            torch.nn.ELU(),
            SoundStreamDecoderBlock(out_channels=8*C, stride=2),
            torch.nn.ELU(),
            SoundStreamDecoderBlock(out_channels=4*C, stride=4),
            torch.nn.ELU(),
            SoundStreamDecoderBlock(out_channels=2*C, stride=5),
            torch.nn.ELU(),
            SoundStreamDecoderBlock(out_channels=C, stride=7),
            torch.nn.ELU(),
            CausalConv1d(in_channels=C, out_channels=1, kernel_size=7),
            nn.Conv1d(1, 1, 205, padding = 1, stride = 1)
        )

    def forward(self, x):
        y = self.layers(x)
        return y

class SPECDiscriminator(nn.Module):
    def __init__(self, n_samples=253074, sample_rate=16000):
        super(SPECDiscriminator, self).__init__()
        self.spectrogram_transform = Compose([
            MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=128, n_mels=128).to(device),
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
        audio_waveform = audio_waveform.to(device)
        spectrogram = self.spectrogram_transform(audio_waveform)
        out = self.model(spectrogram)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
    
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, autoencoder, spec_discriminator, criterion, adv_criterion):
        super().__init__()
        self.autoencoder = autoencoder
        self.spec_discriminator = spec_discriminator
        self.criterion = criterion
        self.adv_criterion = adv_criterion
        self.mel_transform = Compose([
            MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=128, n_mels=128),
            AmplitudeToDB()
        ])
        self.automatic_optimization = False  # Disable automatic optimization

    def forward(self, x):
        return self.autoencoder(x)

    def training_step(self, batch, batch_idx):
        inputs = batch
        batch_size = inputs.size(0)
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        # Get optimizers
        spec_d_optimizer, ae_optimizer = self.optimizers()

        # Train Discriminator
        spec_d_optimizer.zero_grad()
        real_validity = self.spec_discriminator(inputs)
        spec_d_real_loss = self.adv_criterion(real_validity, real_labels)

        outputs = self.autoencoder(inputs)
        fake_validity = self.spec_discriminator(outputs.detach())
        spec_d_fake_loss = self.adv_criterion(fake_validity, fake_labels)

        spec_d_loss = (spec_d_real_loss + spec_d_fake_loss) / 2
        self.log("spec_d_loss", spec_d_loss)
        self.manual_backward(spec_d_loss)
        spec_d_optimizer.step()

        # Train Autoencoder
        ae_optimizer.zero_grad()
        outputs = self.autoencoder(inputs)
        ae_loss = self.criterion(outputs, inputs)
        fake_validity = self.spec_discriminator(outputs)
        generation_spec_loss = self.adv_criterion(fake_validity, real_labels)
        ae_g_loss = ae_loss + generation_spec_loss
        self.log("ae_loss", ae_loss)
        self.log("generation_spec_loss", generation_spec_loss)
        self.log("ae_g_loss", ae_g_loss)
        self.manual_backward(ae_g_loss)
        ae_optimizer.step()

        if batch_idx % 5 == 0:
            input_audio = inputs.cpu().detach().numpy().astype(np.float32)
            output_audio = outputs.cpu().detach().numpy().astype(np.float32)
            input_spectrogram = self.mel_transform(torch.tensor(input_audio[0][0])).cpu().detach().numpy()
            output_spectrogram = self.mel_transform(torch.tensor(output_audio[0][0])).cpu().detach().numpy()

            self.logger.experiment.log_audio(audio_data=input_audio[0][0], sample_rate=16000, file_name=f'train_batch_{batch_idx}.wav')
            self.logger.experiment.log_audio(audio_data=output_audio[0][0], sample_rate=16000, file_name=f'train_batch_{batch_idx}_recons.wav')
            self.logger.experiment.log_image(image_data=input_spectrogram, name=f'train_batch_{batch_idx}_spec.png')
            self.logger.experiment.log_image(image_data=output_spectrogram, name=f'train_batch_{batch_idx}_recons_spec.png')

        return ae_g_loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        outputs = self(inputs)

        if outputs.shape[2] < inputs.shape[2]:
            padding_needed = inputs.shape[2] - outputs.shape[2]
            outputs = torch.nn.functional.pad(outputs, (0, padding_needed))
        elif outputs.shape[2] > inputs.shape[2]:
            outputs = outputs[:, :, :inputs.shape[2]]

        loss = self.criterion(outputs, inputs)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        input_audio = inputs.cpu().detach().numpy().astype(np.float32)
        output_audio = outputs.cpu().detach().numpy().astype(np.float32)
        input_spectrogram = self.mel_transform(torch.tensor(input_audio[0][0])).cpu().detach().numpy()
        output_spectrogram = self.mel_transform(torch.tensor(output_audio[0][0])).cpu().detach().numpy()

        self.logger.experiment.log_audio(audio_data=input_audio[0][0], sample_rate=16000, file_name=f'val_batch_{batch_idx}.wav')
        self.logger.experiment.log_audio(audio_data=output_audio[0][0], sample_rate=16000, file_name=f'val_batch_{batch_idx}_recons.wav')
        self.logger.experiment.log_image(image_data=input_spectrogram, name=f'val_batch_{batch_idx}_spec.png')
        self.logger.experiment.log_image(image_data=output_spectrogram, name=f'val_batch_{batch_idx}_recons_spec.png')

        return loss

    def configure_optimizers(self):
        ae_optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.0003, betas=(0.5, 0.999))
        spec_d_optimizer = optim.Adam(self.spec_discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))
        return [spec_d_optimizer, ae_optimizer]


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
        if waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif waveform.shape[1] < self.target_length:
            padding_needed = self.target_length - waveform.shape[1]
            waveform = nn.functional.pad(waveform, (0, padding_needed))
        return waveform.to(device)

class RandomAudioDataset(Dataset):
    def __init__(self, num_samples, target_length=84358):
        self.num_samples = num_samples
        self.target_length = target_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        waveform = torch.rand(1, self.target_length).to(device)
        return waveform
    
    
# Setup models, optimizers, and other components
in_dim = 1
h_dim = 64

encoder = SoundStreamEncoder(in_dim, h_dim).to(device)
decoder = SoundStreamDecoder(h_dim, in_dim).to(device)
autoencoder = nn.Sequential(encoder, decoder).to(device)
spec_discriminator = SPECDiscriminator().to(device)

criterion = nn.MSELoss().to(device)
adv_criterion = nn.BCELoss().to(device)

lit_autoencoder = LitAutoEncoder(autoencoder, spec_discriminator, criterion, adv_criterion)

# Load data
data_path = "./RAVDES"
audio_dataset = RAVDESSDataset(data_path)
batch_size = 16
train_size = int(0.9 * len(audio_dataset))
val_size = len(audio_dataset) - train_size
train_dataset, val_dataset = random_split(audio_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Setup Comet Logger
comet_logger = CometLogger(
    api_key="<HIDDEN>",
    project_name="Emotional_speech"
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=20,
    mode='min'
)
# Training
trainer = pl.Trainer(logger=comet_logger, max_epochs=100, callbacks=[early_stopping_callback])
trainer.fit(lit_autoencoder, train_loader, val_loader)
