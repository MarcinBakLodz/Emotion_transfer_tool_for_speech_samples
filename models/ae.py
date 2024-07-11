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

from models.layers import Decoder, Encoder, VectorQuantizer, DualLatentEncoder, DualLatentDecoder
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


class VQVAE(AE):
    def __init__(self, args_dict):
        super(VQVAE, self).__init__(args_dict)

        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_e=self.n_e, e_dim=self.latent_dim, beta=self.beta, check_dims=self.verbose)

    def configure_optimizers(self):
        self.vector_quantization.set_device(self.device)
        return super().configure_optimizers()

    def forward(self, x):
        z_e = self.encoder(x)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, embedding_loss, perplexity

    def training_step(self, batch, batch_idx):
        x_hat, embedding_loss, perplexity = self(batch)
        recon_loss = self.loss_fn(x_hat, batch)

        self.log_training_step_metrics(recon_loss, embedding_loss, perplexity, x_hat, batch, batch_idx)

        return recon_loss + embedding_loss

    def validation_step(self, batch, batch_idx):
        x_hat, embedding_loss, perplexity = self(batch)
        recon_loss = self.loss_fn(x_hat, batch)

        self.log_validation_step_metrics(recon_loss, embedding_loss, perplexity, x_hat, batch, batch_idx)

        return recon_loss + embedding_loss

    @torch.no_grad()
    def log_training_step_metrics(self, recon_loss, embedding_loss, perplexity, x_hat, batch, batch_idx):
        super().log_training_step_metrics(recon_loss + embedding_loss, x_hat, batch, batch_idx)

        # Log additional metrics specific to VQVAE
        self.log('train_reconstruction_loss', recon_loss, sync_dist=True, batch_size=self.batch_size)
        self.log('train_embedding_loss', embedding_loss, sync_dist=True, batch_size=self.batch_size)
        self.log('train_perplexity', perplexity, sync_dist=True, batch_size=self.batch_size)

    @torch.no_grad()
    @skip_if_sanity_checking
    def log_validation_step_metrics(self, recon_loss, embedding_loss, perplexity, x_hat, batch, batch_idx):
        super().log_validation_step_metrics(recon_loss + embedding_loss, x_hat, batch, batch_idx)

        self.log('val_reconstruction_loss', recon_loss, sync_dist=True, batch_size=self.batch_size)
        self.log('val_embedding_loss', embedding_loss, sync_dist=True, batch_size=self.batch_size)
        self.log('val_perplexity', perplexity, sync_dist=True, batch_size=self.batch_size)


class DualLatentAE(VQVAE):
    def __init__(self, args_dict):
        super(DualLatentAE, self).__init__(args_dict)
        del self.vector_quantization

        self.encoder = DualLatentEncoder(in_dim=self.in_dim,
                                         h_dim=self.h_dim,
                                         cont_latent_dim=self.cont_latent_dim,
                                         vq_latent_dim=self.vq_latent_dim,
                                         n_e=self.n_e,
                                         beta=self.beta,
                                         verbose=self.verbose)

        self.decoder = DualLatentDecoder(in_dim=self.latent_dim, h_dim=self.h_dim)

    def configure_optimizers(self):
        self.encoder.vector_quantization[1].set_device(self.device)
        return AE.configure_optimizers(self)

    def forward(self, x):
        x, z_q, embedding_loss, perplexity = self.encoder(x)
        x_hat = self.decoder(x, z_q)
        return x_hat, embedding_loss, perplexity
