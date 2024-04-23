from natsort import os_sorted
import numpy as np
import os
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchvision.transforms import Compose
from torchmetrics.audio import PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility
from torchmetrics.image import StructuralSimilarityIndexMeasure

from models.layers import Decoder, Encoder, SoundStreamEncoder, SoundStreamDecoder
from models.model_utils import skip_if_sanity_checking


class AE(LightningModule):
    def __init__(self, args_dict):
        super().__init__()
        self.__dict__.update(args_dict)

        self.encoder = Encoder(in_dim=self.in_dim, h_dim=self.h_dim, latent_dim=self.latent_dim)
        self.decoder = Decoder(in_dim=self.latent_dim, h_dim=self.h_dim)

        self.mel_transform = Compose([
            MelSpectrogram(sample_rate=self.sr, n_fft=1024, hop_length=128, n_mels=128),
            AmplitudeToDB()
        ])

        # loss function
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

        # metrics
        self.pesq = PerceptualEvaluationSpeechQuality(fs=self.sr, mode='wb', n_processes=os.cpu_count())  # -> https://lightning.ai/docs/torchmetrics/stable/audio/perceptual_evaluation_speech_quality.html
        self.stoi = ShortTimeObjectiveIntelligibility(fs=self.sr, extended=True)  # -> https://lightning.ai/docs/torchmetrics/stable/audio/short_time_objective_intelligibility.html
        self.ssim = StructuralSimilarityIndexMeasure()  # -> https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html

    def configure_optimizers(self):
        # cant call .to(device) on Compose, has to be called on individual modules inside (theoretically it should move automatically, but still it creates window on CPU)
        # we're doing it in configure_optimizers() since self.device is already known
        for transform in self.mel_transform.transforms:
            if isinstance(transform, torch.nn.Module):
                transform.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, x):
        z_e = self.encoder(x)
        x_hat = self.decoder(z_e)
        return x_hat

    def training_step(self, batch, batch_idx):
        x_hat = self(batch)
        recon_loss = self.loss_fn(x_hat, batch)

        self.log_training_step_metrics(recon_loss, x_hat, batch, batch_idx)

        return recon_loss

    def validation_step(self, batch, batch_idx):
        x_hat = self(batch)
        recon_loss = self.loss_fn(x_hat, batch)

        self.log_validation_step_metrics(recon_loss, x_hat, batch, batch_idx)

        return recon_loss

    def on_fit_end(self):
        self.log_best_checkpoint()

    @torch.no_grad()
    def log_training_step_metrics(self, recon_loss, x_hat, batch, batch_idx):
        # log loss
        self.log('train_loss', recon_loss, sync_dist=True, batch_size=self.batch_size)

        # log image metrics
        self.log('train_ssim', self.ssim(self.mel_transform(x_hat), self.mel_transform(batch)), sync_dist=True, batch_size=self.batch_size)

        # log audio metrics every 10th batch, since they're calculated on CPU and it takes some time
        if batch_idx % 10 == 0:
            for metric_name, metric_func in [('train_pesq', self.pesq), ('train_stoi', self.stoi)]:
                try:
                    self.log(metric_name, metric_func(x_hat, batch), sync_dist=True, batch_size=self.batch_size)
                except TypeError:
                    continue

        # log exemplary data, let's save 5 examples per epoch
        if batch_idx < 5:
            self.logger.experiment.log_audio(audio_data=batch[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}.wav')
            self.logger.experiment.log_audio(audio_data=x_hat[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_recons.wav')
            self.logger.experiment.log_image(image_data=self.mel_transform(batch[0]).to('cpu').numpy(), image_channels='first', name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_target')
            self.logger.experiment.log_image(image_data=self.mel_transform(x_hat[0]).to('cpu').numpy(), image_channels='first', name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_pred')

    @torch.no_grad()
    @skip_if_sanity_checking
    def log_validation_step_metrics(self, recon_loss, x_hat, batch, batch_idx):
        # log loss
        self.log('val_loss', recon_loss, sync_dist=True, batch_size=self.batch_size)

        # log image metrics
        self.log('val_ssim', self.ssim(self.mel_transform(x_hat), self.mel_transform(batch)), sync_dist=True, batch_size=self.batch_size)

        # log audio metrics every 10th batch, since they're calculated on CPU and it takes some time
        if batch_idx % 10 == 0:
            for metric_name, metric_func in [('val_pesq', self.pesq), ('val_stoi', self.stoi)]:
                try:
                    self.log(metric_name, metric_func(x_hat, batch), sync_dist=True, batch_size=self.batch_size)
                except TypeError:
                    continue

        # log exemplary data, let's save 5 examples per epoch
        if batch_idx < 5:
            self.logger.experiment.log_audio(audio_data=batch[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}.wav')
            self.logger.experiment.log_audio(audio_data=x_hat[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_recons.wav')
            self.logger.experiment.log_image(image_data=self.mel_transform(batch[0]).to('cpu').numpy(), image_channels='first', name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_target')
            self.logger.experiment.log_image(image_data=self.mel_transform(x_hat[0]).to('cpu').numpy(), image_channels='first', name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_pred')

    @rank_zero_only  # needed for self.logger.experiment.get_key() to work properly when using >1 GPUs
    def log_best_checkpoint(self):
        best_checkpoint = os_sorted(os.listdir(f'../results/{self.logger.experiment.get_key()}/checkpoints'))[-1]
        self.logger.experiment.log_asset(f'../results/{self.logger.experiment.get_key()}/checkpoints/{best_checkpoint}')


class SoundStreamAE(AE):
    def __init__(self, args_dict):
        super().__init__(args_dict)

        self.encoder = SoundStreamEncoder(C=self.h_dim, D=self.latent_dim)
        self.decoder = SoundStreamDecoder(C=self.h_dim, D=self.latent_dim)
