from itertools import chain
from natsort import os_sorted
import numpy as np
import os
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchvision.transforms import Compose
import torchvision
from torchmetrics.audio import PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility
from torchmetrics.image import StructuralSimilarityIndexMeasure

from models.layers import *
from models.layers_utils import crop_random_patch
from models.model_utils import skip_if_sanity_checking, create_marker_file


class AE(LightningModule):
    def __init__(self, args_dict):
        super().__init__()
        self.__dict__.update(args_dict)

        self.automatic_optimization = False

        self.encoder = Encoder(in_dim=self.in_dim, h_dim=self.h_dim, latent_dim=self.latent_dim)
        self.decoder = Decoder(in_dim=self.latent_dim, h_dim=self.h_dim)

        self.wave_discriminator = torch.nn.Sequential(
            Encoder(in_dim=self.in_dim, h_dim=self.wave_disc_h_dim, latent_dim=self.latent_dim),
            torch.nn.AdaptiveAvgPool1d(1),  # Reduce feature map size to 1
            torch.nn.Flatten()  # Remove extra dimensions
        )

        # if self.use_stft_disc:
        #     self.stft_discriminator = STFTDiscriminator(h_dim=self.stft_disc_h_dim)

        self.mel_transform = Compose([
            MelSpectrogram(sample_rate=self.sr, n_fft=1024, hop_length=128, n_mels=128),
            AmplitudeToDB()
        ])

        # loss function
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.disc_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

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

        # Generator optimizer
        optimizer_g = torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.g_learning_rate, weight_decay=self.weight_decay)

        # Discriminator optimizer
        discriminator_params = []
        if self.use_wave_disc:
            discriminator_params.append({'params': self.wave_discriminator.parameters(), 'lr': self.d_wave_learning_rate})
        if self.use_stft_disc:
            discriminator_params.append({'params': self.stft_discriminator.parameters(), 'lr': self.d_stft_learning_rate})

        optimizers = [optimizer_g]
        if discriminator_params:
            optimizer_d = torch.optim.Adam(discriminator_params, weight_decay=self.weight_decay)
            optimizers.append(optimizer_d)

        return optimizers, []

    def forward(self, x):
        z_e = self.encoder(x)
        x_hat = self.decoder(z_e)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch  # label not needed
        optimizers = self.optimizers()

        # Assuming the first optimizer is always the generator optimizer
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        optimizer_g = optimizers[0]

        # Check if there is a discriminator optimizer
        optimizer_d = optimizers[1] if len(optimizers) > 1 else None

        # train generator
        self.toggle_optimizer(optimizer_g)
        x_hat = self(x)
        recon_loss = self.loss_fn(x_hat, x)

        g_wave_loss, g_stft_loss = 0, 0

        if self.use_wave_disc:
            g_output_fake_wave = self.wave_discriminator(x_hat)
            g_wave_loss = self.disc_loss_fn(g_output_fake_wave, torch.ones_like(g_output_fake_wave))

        if self.use_stft_disc:
            g_output_fake_stft = self.stft_discriminator(self.mel_transform(x_hat))
            g_stft_loss = self.disc_loss_fn(g_output_fake_stft, torch.ones_like(g_output_fake_stft))

        g_loss = (self.alpha * recon_loss) + ((g_wave_loss + g_stft_loss) / 2)

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        d_wave_loss, d_stft_loss, r1_wave_penalty, r1_stft_penalty = 0, 0, 0, 0
        if optimizer_d:
            x_hat = self(x)
            mel_x, mel_x_hat = self.mel_transform(x), self.mel_transform(x_hat)
            self.toggle_optimizer(optimizer_d)

            if self.use_wave_disc:
                d_output_real_wave = self.wave_discriminator(x)
                d_output_fake_wave = self.wave_discriminator(x_hat)

                d_wave_loss_real = self.disc_loss_fn(d_output_real_wave, torch.ones_like(d_output_real_wave))
                d_wave_loss_fake = self.disc_loss_fn(d_output_fake_wave, torch.zeros_like(d_output_fake_wave))
                r1_wave_penalty = self.compute_R1_loss(x, discriminator_type='wave')
                d_wave_loss = (d_wave_loss_real + d_wave_loss_fake) / 2 + r1_wave_penalty

            if self.use_stft_disc:
                d_output_real_stft = self.stft_discriminator(mel_x_hat)
                d_output_fake_stft = self.stft_discriminator(mel_x)

                d_stft_loss_real = self.disc_loss_fn(d_output_real_stft, torch.ones_like(d_output_real_stft))
                d_stft_loss_fake = self.disc_loss_fn(d_output_fake_stft, torch.zeros_like(d_output_fake_stft))
                r1_stft_penalty = self.compute_R1_loss(mel_x, discriminator_type='stft')
                d_stft_loss = (d_stft_loss_real + d_stft_loss_fake) / 2 + r1_stft_penalty

            d_loss = (d_wave_loss + d_stft_loss) / 2

            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

        self.log_step_metrics(recon_loss, g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss, r1_wave_penalty, r1_stft_penalty, x_hat, x, batch_idx, step_name='train')

    def validation_step(self, batch, batch_idx):
        x, _ = batch  # label not needed
        x_hat = self(x)
        mel_x, mel_x_hat = self.mel_transform(x), self.mel_transform(x_hat)

        recon_loss = self.loss_fn(x_hat, x)

        g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss = 0, 0, 0, 0

        if self.use_wave_disc:
            g_output_fake_wave = self.wave_discriminator(x_hat)
            g_wave_loss = self.disc_loss_fn(g_output_fake_wave, torch.ones_like(g_output_fake_wave))

            d_output_real_wave = self.wave_discriminator(x)
            d_output_fake_wave = self.wave_discriminator(x_hat)
            d_wave_loss_real = self.disc_loss_fn(d_output_real_wave, torch.ones_like(d_output_real_wave))
            d_wave_loss_fake = self.disc_loss_fn(d_output_fake_wave, torch.zeros_like(d_output_fake_wave))
            d_wave_loss = (d_wave_loss_real + d_wave_loss_fake) / 2

        if self.use_stft_disc:
            g_output_fake_stft = self.stft_discriminator(mel_x_hat)
            g_stft_loss = self.disc_loss_fn(g_output_fake_stft, torch.ones_like(g_output_fake_stft))

            d_output_real_stft = self.stft_discriminator(mel_x_hat)
            d_output_fake_stft = self.stft_discriminator(mel_x)
            d_stft_loss_real = self.disc_loss_fn(d_output_real_stft, torch.ones_like(d_output_real_stft))
            d_stft_loss_fake = self.disc_loss_fn(d_output_fake_stft, torch.zeros_like(d_output_fake_stft))
            d_stft_loss = (d_stft_loss_real + d_stft_loss_fake) / 2

        self.log_step_metrics(recon_loss, g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss, 0, 0, x_hat, x, batch_idx, step_name='val')

    def compute_R1_loss(self, x, discriminator_type):
        discriminators = {'wave': getattr(self, 'wave_discriminator', None), 'stft': getattr(self, 'stft_discriminator', None)}
        if discriminator_type not in discriminators or discriminators[discriminator_type] is None:
            raise ValueError('Invalid or unavailable discriminator_type in R1 loss calculation. Should be one of: ["wave", "stft"].')

        gammas = {'wave': getattr(self, 'gamma_wave', 0), 'stft': getattr(self, 'gamma_stft', 0)}
        gamma = gammas[discriminator_type]
        if gamma <= 0:
            return 0

        x.requires_grad_()
        pred_real = discriminators[discriminator_type](x).sum()
        grad_real = torch.autograd.grad(outputs=pred_real, inputs=x, create_graph=True, retain_graph=True)[0]
        grad_penalty = torch.sum(grad_real.pow(2), dim=list(range(1, grad_real.ndim)))
        return grad_penalty.mean() * (gamma * 0.5)

    def on_fit_end(self):
        self.log_best_checkpoint()

    @torch.no_grad()
    @skip_if_sanity_checking
    def log_step_metrics(self, recon_loss, g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss, x_hat, x, batch_idx, step_name):
        # log loss
        self.log(f'{step_name}_g_recons_loss', recon_loss, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{step_name}_g_wave_loss', g_wave_loss, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{step_name}_g_stft_loss', g_stft_loss, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{step_name}_d_wave_loss', d_wave_loss, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{step_name}_d_stft_loss', d_stft_loss, sync_dist=True, batch_size=self.batch_size)

        # if step_name == 'train':
        #     self.log(f'{step_name}_r1_wave_penalty', r1_wave_penalty, sync_dist=True, batch_size=self.batch_size)
        #     self.log(f'{step_name}_r1_stft_penalty', r1_stft_penalty, sync_dist=True, batch_size=self.batch_size)

        # log image metrics
        self.log(f'{step_name}_ssim', self.ssim(self.mel_transform(x_hat), self.mel_transform(x)), sync_dist=True, batch_size=self.batch_size)

        # log audio metrics every 10th batch, since they're calculated on CPU and it takes some time
        if batch_idx % 10 == 0:
            for metric_name, metric_func in [(f'{step_name}_pesq', self.pesq), (f'{step_name}_stoi', self.stoi)]:
                try:
                    self.log(metric_name, metric_func(x_hat, x), sync_dist=True, batch_size=self.batch_size)
                except TypeError:
                    continue

        # log exemplary data, let's save 5 examples per epoch
        if batch_idx < 5:
            self.logger.experiment.log_audio(audio_data=x[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'{step_name}_epoch_{self.trainer.current_epoch}_{batch_idx}.wav')
            self.logger.experiment.log_audio(audio_data=x_hat[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'{step_name}_epoch_{self.trainer.current_epoch}_{batch_idx}_recons.wav')
            self.logger.experiment.log_image(image_data=self.mel_transform(x[0]).to('cpu').numpy(), image_channels='first', name=f'{step_name}_epoch_{self.trainer.current_epoch}_{batch_idx}_target')
            self.logger.experiment.log_image(image_data=self.mel_transform(x_hat[0]).to('cpu').numpy(), image_channels='first', name=f'{step_name}_epoch_{self.trainer.current_epoch}_{batch_idx}_pred')

    @rank_zero_only  # needed for self.logger.experiment.get_key() to work properly when using >1 GPUs
    def log_best_checkpoint(self):
        best_checkpoint = os_sorted(os.listdir(f'{self.results_root}/{self.logger.experiment.get_key()}/checkpoints'))[-1]
        self.logger.experiment.log_asset(f'{self.results_root}/{self.logger.experiment.get_key()}/checkpoints/{best_checkpoint}')
        create_marker_file(f'{self.results_root}/{self.logger.experiment.get_key()}')



class VQVAE(AE):
    def __init__(self, args_dict):
        super(VQVAE, self).__init__(args_dict)

        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_e=self.n_e, e_dim=self.latent_dim, beta=self.beta)

    def configure_optimizers(self):
        for transform in self.mel_transform.transforms:
            if isinstance(transform, torch.nn.Module):
                transform.to(self.device)

        self.vector_quantization.set_device(self.device)
        optimizer_g = torch.optim.Adam(chain(self.encoder.parameters(), self.vector_quantization.parameters(), self.decoder.parameters()), lr=self.g_learning_rate, weight_decay=self.weight_decay)
        optimizer_d = torch.optim.Adam(self.wave_discriminator.parameters(), lr=self.d_learning_rate, weight_decay=self.weight_decay)

        return [optimizer_g, optimizer_d], []

    def forward(self, x):
        z_e = self.encoder(x)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, embedding_loss, perplexity

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()

        x_hat, embedding_loss, perplexity = self(batch)
        embedding_loss *= 100

        # train generator
        self.toggle_optimizer(optimizer_g)
        recon_loss = self.loss_fn(x_hat, batch) * 100

        g_output_fake = self.wave_discriminator(x_hat)
        g_wave_loss = self.disc_loss_fn(g_output_fake, torch.ones_like(g_output_fake))
        g_loss = recon_loss + embedding_loss + g_wave_loss

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        x_hat, *_ = self(batch)
        self.toggle_optimizer(optimizer_d)

        d_output_real = self.wave_discriminator(batch)
        d_output_fake = self.wave_discriminator(x_hat)

        d_wave_loss_real = self.disc_loss_fn(d_output_real, torch.ones_like(d_output_real))
        d_wave_loss_fake = self.disc_loss_fn(d_output_fake, torch.zeros_like(d_output_fake))

        d_wave_loss = (d_wave_loss_real + d_wave_loss_fake) / 2

        self.manual_backward(d_wave_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        self.log_training_step_metrics(recon_loss, g_wave_loss, d_wave_loss, embedding_loss, perplexity, x_hat, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        x_hat, embedding_loss, perplexity = self(batch)
        recon_loss = self.loss_fn(x_hat, batch)

        g_output_fake = self.wave_discriminator(x_hat)
        g_wave_loss = self.disc_loss_fn(g_output_fake, torch.ones_like(g_output_fake))

        d_output_real = self.wave_discriminator(batch)
        d_output_fake = self.wave_discriminator(x_hat)

        d_wave_loss_real = self.disc_loss_fn(d_output_real, torch.ones_like(d_output_real))
        d_wave_loss_fake = self.disc_loss_fn(d_output_fake, torch.zeros_like(d_output_fake))

        d_wave_loss = (d_wave_loss_real + d_wave_loss_fake) / 2

        self.log_validation_step_metrics(recon_loss, g_wave_loss, d_wave_loss, embedding_loss, perplexity, x_hat, batch, batch_idx)

    @torch.no_grad()
    def log_training_step_metrics(self, recon_loss, g_wave_loss, d_wave_loss, embedding_loss, perplexity, x_hat, batch, batch_idx):
        super().log_training_step_metrics(recon_loss, g_wave_loss, d_wave_loss, x_hat, batch, batch_idx)

        # Log additional metrics specific to VQVAE
        self.log('train_embedding_loss', embedding_loss, sync_dist=True, batch_size=self.batch_size)
        self.log('train_perplexity', perplexity, sync_dist=True, batch_size=self.batch_size)

    @torch.no_grad()
    @skip_if_sanity_checking
    def log_validation_step_metrics(self, recon_loss, g_wave_loss, d_wave_loss, embedding_loss, perplexity, x_hat, batch, batch_idx):
        super().log_validation_step_metrics(recon_loss, g_wave_loss, d_wave_loss, x_hat, batch, batch_idx)

        self.log('val_embedding_loss', embedding_loss, sync_dist=True, batch_size=self.batch_size)
        self.log('val_perplexity', perplexity, sync_dist=True, batch_size=self.batch_size)


# class DualLatentAE(VQVAE):
#     def __init__(self, args_dict):
#         super().__init__(args_dict)
#         if self.cont_latent_dim + self.vq_latent_dim != self.latent_dim:
#             raise ValueError(f"Erroneous config! cont_latent_dim + vq_latent_dim must be equal to latent_dim.")
#         del self.vector_quantization

#         self.encoder = DualLatentEncoder(in_dim=self.in_dim,
#                                          h_dim=self.h_dim,
#                                          cont_latent_dim=self.cont_latent_dim,
#                                          vq_latent_dim=self.vq_latent_dim,
#                                          n_e=self.n_e,
#                                          beta=self.beta)

#         self.decoder = DualLatentDecoder(in_dim=self.latent_dim, h_dim=self.h_dim)

#         if self.use_classifier:
#             self.classifier = Classifier(in_dim=self.cont_latent_dim, h_dim=self.classifier_h_dim, n_classes=self.n_classes, fc_mult=16)

#         if self.use_grl_classifier:
#             self.grl_classifier = torch.nn.Sequential(
#                 GradientReversal(alpha=self.grl_alpha),
#                 Classifier(in_dim=self.vq_latent_dim, h_dim=self.classifier_h_dim, n_classes=self.n_classes, fc_mult=16)
#             )

#         # loss function
#         self.classifier_loss_fn = torch.nn.CrossEntropyLoss()

#         # metrics
#         self.accuracy = MulticlassAccuracy(num_classes=self.n_classes)
#         self.auroc = MulticlassAUROC(num_classes=self.n_classes, average='macro')
#         self.f1 = MulticlassF1Score(num_classes=self.n_classes, top_k=1, average='micro', multidim_average='global')

#     def configure_optimizers(self):
#         self.encoder.vector_quantization[2].set_device(self.device)

#         for transform in self.mel_transform.transforms:
#             if isinstance(transform, torch.nn.Module):
#                 transform.to(self.device)

#         optimizer_g_params = [
#             {'params': self.encoder.parameters(), 'lr': self.g_enc_learning_rate},
#             {'params': self.decoder.parameters(), 'lr': self.g_dec_learning_rate}
#         ]

#         if self.use_classifier:
#             optimizer_g_params.append({'params': self.classifier.parameters(), 'lr': self.g_clf_learning_rate})
#         if self.use_grl_classifier:
#             optimizer_g_params.append({'params': self.grl_classifier.parameters(), 'lr': self.g_grl_learning_rate})

#         optimizer_g = torch.optim.Adam(optimizer_g_params, weight_decay=self.weight_decay)

#         # Discriminator optimizer
#         discriminator_params = []
#         if self.use_wave_disc:
#             discriminator_params.append({'params': self.wave_discriminator.parameters(), 'lr': self.d_wave_learning_rate})
#         if self.use_stft_disc:
#             discriminator_params.append({'params': self.stft_discriminator.parameters(), 'lr': self.d_stft_learning_rate})

#         optimizers = [optimizer_g]
#         if discriminator_params:
#             optimizer_d = torch.optim.Adam(discriminator_params, weight_decay=self.weight_decay)
#             optimizers.append(optimizer_d)

#         return optimizers, []

#     def forward(self, x):
#         x, z_q, embedding_loss, perplexity = self.encoder(x)
#         x_hat = self.decoder(x, z_q)
#         y_pred = self.classifier(x) if self.use_classifier else None
#         y_pred_grl = self.grl_classifier(z_q) if self.use_grl_classifier else None
#         return x_hat, embedding_loss, perplexity, y_pred, y_pred_grl

#     def training_step(self, batch, batch_idx):
#         x, y_true = batch
#         optimizers = self.optimizers()

#         # Assuming the first optimizer is always the generator optimizer
#         if not isinstance(optimizers, list):
#             optimizers = [optimizers]
#         optimizer_g = optimizers[0]

#         # Check if there is a discriminator optimizer
#         optimizer_d = optimizers[1] if len(optimizers) > 1 else None

#         # train generator
#         self.toggle_optimizer(optimizer_g)
#         x_hat, embedding_loss, perplexity, y_pred, y_pred_grl = self(x)
#         recon_loss = self.loss_fn(x_hat, x)
#         classification_loss = self.classifier_loss_fn(y_pred, y_true) if self.use_classifier else 0
#         grl_classification_loss = self.classifier_loss_fn(y_pred_grl, y_true) if self.use_grl_classifier else 0

#         g_wave_loss, g_stft_loss = 0, 0
#         if self.use_wave_disc:
#             g_output_fake_wave = self.wave_discriminator(x_hat)
#             g_wave_loss = self.disc_loss_fn(g_output_fake_wave, torch.ones_like(g_output_fake_wave))

#         if self.use_stft_disc:
#             g_output_fake_stft = self.stft_discriminator(self.mel_transform(x_hat))
#             g_stft_loss = self.disc_loss_fn(g_output_fake_stft, torch.ones_like(g_output_fake_stft))

#         g_loss = (self.alpha * (recon_loss + embedding_loss)) + (self.delta * classification_loss) + (self.delta * grl_classification_loss) + ((g_wave_loss + g_stft_loss) / 2)

#         self.manual_backward(g_loss)
#         optimizer_g.step()
#         optimizer_g.zero_grad()
#         self.untoggle_optimizer(optimizer_g)

#         # train discriminator
#         d_wave_loss, d_stft_loss, r1_wave_penalty, r1_stft_penalty = 0, 0, 0, 0
#         if optimizer_d:
#             x_hat, *_ = self(x)
#             mel_x, mel_x_hat = self.mel_transform(x), self.mel_transform(x_hat)
#             self.toggle_optimizer(optimizer_d)

#             if self.use_wave_disc:
#                 d_output_real_wave = self.wave_discriminator(x)
#                 d_output_fake_wave = self.wave_discriminator(x_hat)

#                 d_wave_loss_real = self.disc_loss_fn(d_output_real_wave, torch.ones_like(d_output_real_wave))
#                 d_wave_loss_fake = self.disc_loss_fn(d_output_fake_wave, torch.zeros_like(d_output_fake_wave))
#                 r1_wave_penalty = self.compute_R1_loss(x, discriminator_type='wave')
#                 d_wave_loss = (d_wave_loss_real + d_wave_loss_fake) / 2 + r1_wave_penalty

#             if self.use_stft_disc:
#                 d_output_real_stft = self.stft_discriminator(mel_x_hat)
#                 d_output_fake_stft = self.stft_discriminator(mel_x)

#                 d_stft_loss_real = self.disc_loss_fn(d_output_real_stft, torch.ones_like(d_output_real_stft))
#                 d_stft_loss_fake = self.disc_loss_fn(d_output_fake_stft, torch.zeros_like(d_output_fake_stft))
#                 r1_stft_penalty = self.compute_R1_loss(mel_x, discriminator_type='stft')
#                 d_stft_loss = (d_stft_loss_real + d_stft_loss_fake) / 2 + r1_stft_penalty

#             d_loss = (d_wave_loss + d_stft_loss) / 2

#             self.manual_backward(d_loss)
#             optimizer_d.step()
#             optimizer_d.zero_grad()
#             self.untoggle_optimizer(optimizer_d)

#         self.log_step_metrics(recon_loss, classification_loss, grl_classification_loss, g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss, embedding_loss, perplexity, r1_wave_penalty, r1_stft_penalty, x_hat, y_pred, y_pred_grl, x, y_true, batch_idx, step_name='train')

#     def validation_step(self, batch, batch_idx):
#         x, y_true = batch
#         x_hat, embedding_loss, perplexity, y_pred, y_pred_grl = self(x)
#         mel_x, mel_x_hat = self.mel_transform(x), self.mel_transform(x_hat)

#         recon_loss = self.loss_fn(x_hat, x)
#         classification_loss = self.classifier_loss_fn(y_pred, y_true) if self.use_classifier else 0
#         grl_classification_loss = self.classifier_loss_fn(y_pred_grl, y_true) if self.use_grl_classifier else 0

#         g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss = 0, 0, 0, 0

#         if self.use_wave_disc:
#             g_output_fake_wave = self.wave_discriminator(x_hat)
#             g_wave_loss = self.disc_loss_fn(g_output_fake_wave, torch.ones_like(g_output_fake_wave))

#             d_output_real_wave = self.wave_discriminator(x)
#             d_output_fake_wave = self.wave_discriminator(x_hat)
#             d_wave_loss_real = self.disc_loss_fn(d_output_real_wave, torch.ones_like(d_output_real_wave))
#             d_wave_loss_fake = self.disc_loss_fn(d_output_fake_wave, torch.zeros_like(d_output_fake_wave))
#             d_wave_loss = (d_wave_loss_real + d_wave_loss_fake) / 2

#         if self.use_stft_disc:
#             g_output_fake_stft = self.stft_discriminator(mel_x_hat)
#             g_stft_loss = self.disc_loss_fn(g_output_fake_stft, torch.ones_like(g_output_fake_stft))

#             d_output_real_stft = self.stft_discriminator(mel_x_hat)
#             d_output_fake_stft = self.stft_discriminator(mel_x)
#             d_stft_loss_real = self.disc_loss_fn(d_output_real_stft, torch.ones_like(d_output_real_stft))
#             d_stft_loss_fake = self.disc_loss_fn(d_output_fake_stft, torch.zeros_like(d_output_fake_stft))
#             d_stft_loss = (d_stft_loss_real + d_stft_loss_fake) / 2

#         self.log_step_metrics(recon_loss, classification_loss, grl_classification_loss, g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss, embedding_loss, perplexity, 0, 0, x_hat, y_pred, y_pred_grl, x, y_true, batch_idx, step_name='val')

#     @torch.no_grad()
#     @skip_if_sanity_checking
#     def log_step_metrics(self, recon_loss, classification_loss, grl_classification_loss, g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss, embedding_loss, perplexity, r1_wave_penalty, r1_stft_penalty, x_hat, y_pred, y_pred_grl, x, y_true, batch_idx, step_name):
#         super().log_step_metrics(recon_loss, g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss, embedding_loss, perplexity, r1_wave_penalty, r1_stft_penalty, x_hat, x, batch_idx, step_name)

#         self.log(f'{step_name}_classification_loss', classification_loss, sync_dist=True, batch_size=self.batch_size)
#         self.log(f'{step_name}_grl_classification_loss', grl_classification_loss, sync_dist=True, batch_size=self.batch_size)

#         # log classification metrics
#         if y_pred is not None:
#             self.log(f'{step_name}_accuracy', self.accuracy(y_pred, y_true), sync_dist=True, batch_size=self.batch_size)
#             self.log(f'{step_name}_auroc', self.auroc(y_pred, y_true), sync_dist=True, batch_size=self.batch_size)
#             self.log(f'{step_name}_f1_score', self.f1(y_pred, y_true), sync_dist=True, batch_size=self.batch_size)

#         # log grl classification metrics
#         if y_pred_grl is not None:
#             self.log(f'{step_name}_grl_accuracy', self.accuracy(y_pred_grl, y_true), sync_dist=True, batch_size=self.batch_size)
#             self.log(f'{step_name}_grl_auroc', self.auroc(y_pred_grl, y_true), sync_dist=True, batch_size=self.batch_size)
#             self.log(f'{step_name}_grl_f1_score', self.f1(y_pred_grl, y_true), sync_dist=True, batch_size=self.batch_size)



class ConditionalGANAE(VQVAE):
    def __init__(self, args_dict):
        super(ConditionalGANAE, self).__init__(args_dict)
        del self.vector_quantization

        self.encoder = DualLatentEncoder(in_dim=self.in_dim,
                                         h_dim=self.h_dim,
                                         cont_latent_dim=self.cont_latent_dim, 
                                         vq_latent_dim=self.vq_latent_dim,
                                         n_e=self.n_e,
                                         beta=self.beta)   
        
        self.decoder = ConditionalDualLatentDecoder(in_dim=1, h_dim=self.h_dim)

    def configure_optimizers(self):
        self.encoder.vector_quantization[1].set_device(self.device)
        for transform in self.mel_transform.transforms:
            if isinstance(transform, torch.nn.Module):
                transform.to(self.device)

        optimizer_g = torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.g_learning_rate, weight_decay=self.weight_decay)
        optimizer_d = torch.optim.Adam(chain(self.wave_discriminator.parameters()), lr=self.d_learning_rate, weight_decay=self.weight_decay)
        # optimizer_cooc = torch.optim.Adam(chain(self.patch_encoder.parameters(), self.cooc_discriminator.parameters()), lr=self.d_learning_rate, weight_decay=self.weight_decay)
        return [optimizer_g, optimizer_d], []
        
    def forward(self, x, label):
        x, z_q, embedding_loss, perplexity = self.encoder(x)
        x_hat = self.decoder(x, z_q, label)
        return x_hat, embedding_loss, perplexity, z_q
    
    def training_step(self, batch, batch_idx):
        x, y_true = batch
        optimizers = self.optimizers()

        # Assuming the first optimizer is always the generator optimizer
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        optimizer_g = optimizers[0]

        # Check if there is a discriminator optimizer
        optimizer_d = optimizers[1] if len(optimizers) > 1 else None

        # train generator
        self.toggle_optimizer(optimizer_g)
        x_hat, embedding_loss, perplexity, z_q = self(x)
        recon_loss = self.loss_fn(x_hat, x)

        g_wave_loss, g_stft_loss = 0, 0
        if self.use_wave_disc:
            g_output_fake_wave = self.wave_discriminator(x_hat)
            g_wave_loss = self.disc_loss_fn(g_output_fake_wave, torch.ones_like(g_output_fake_wave))

        if self.use_stft_disc:
            g_output_fake_stft = self.stft_discriminator(self.mel_transform(x_hat))
            g_stft_loss = self.disc_loss_fn(g_output_fake_stft, torch.ones_like(g_output_fake_stft))

        g_loss = ((recon_loss + embedding_loss)) + ((g_wave_loss + g_stft_loss) / 2)

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # d_wave_loss, d_stft_loss, r1_wave_penalty, r1_stft_penalty = 0, 0, 0, 0
        d_wave_loss, d_stft_loss = 0, 0
        if optimizer_d:
            x_hat, embedding_loss, perplexity, z_q = self(x)
            mel_x, mel_x_hat = self.mel_transform(x), self.mel_transform(x_hat)
            self.toggle_optimizer(optimizer_d)

            if self.use_wave_disc:
                d_output_real_wave = self.wave_discriminator(x)
                d_output_fake_wave = self.wave_discriminator(x_hat)

                d_wave_loss_real = self.disc_loss_fn(d_output_real_wave, torch.ones_like(d_output_real_wave))
                d_wave_loss_fake = self.disc_loss_fn(d_output_fake_wave, torch.zeros_like(d_output_fake_wave))
                # r1_wave_penalty = self.compute_R1_loss(x, discriminator_type='wave')
                d_wave_loss = (d_wave_loss_real + d_wave_loss_fake) / 2 # + r1_wave_penalty

            if self.use_stft_disc:
                d_output_real_stft = self.stft_discriminator(mel_x_hat)
                d_output_fake_stft = self.stft_discriminator(mel_x)

                d_stft_loss_real = self.disc_loss_fn(d_output_real_stft, torch.ones_like(d_output_real_stft))
                d_stft_loss_fake = self.disc_loss_fn(d_output_fake_stft, torch.zeros_like(d_output_fake_stft))
                # r1_stft_penalty = self.compute_R1_loss(mel_x, discriminator_type='stft')
                d_stft_loss = (d_stft_loss_real + d_stft_loss_fake) / 2 # + r1_stft_penalty

            d_loss = (d_wave_loss + d_stft_loss) / 2

            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

        self.log_step_metrics(recon_loss, g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss, embedding_loss, perplexity, x_hat, x, z_q, y_true, batch_idx, step_name='train')

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        x_hat, embedding_loss, perplexity, z_q = self(x, y_true)
        mel_x, mel_x_hat = self.mel_transform(x), self.mel_transform(x_hat)

        recon_loss = self.loss_fn(x_hat, x)

        g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss = 0, 0, 0, 0

        if self.use_wave_disc:
            g_output_fake_wave = self.wave_discriminator(x_hat)
            g_wave_loss = self.disc_loss_fn(g_output_fake_wave, torch.ones_like(g_output_fake_wave))

            d_output_real_wave = self.wave_discriminator(x)
            d_output_fake_wave = self.wave_discriminator(x_hat)
            d_wave_loss_real = self.disc_loss_fn(d_output_real_wave, torch.ones_like(d_output_real_wave))
            d_wave_loss_fake = self.disc_loss_fn(d_output_fake_wave, torch.zeros_like(d_output_fake_wave))
            d_wave_loss = (d_wave_loss_real + d_wave_loss_fake) / 2

        if self.use_stft_disc:
            g_output_fake_stft = self.stft_discriminator(mel_x_hat)
            g_stft_loss = self.disc_loss_fn(g_output_fake_stft, torch.ones_like(g_output_fake_stft))

            d_output_real_stft = self.stft_discriminator(mel_x_hat)
            d_output_fake_stft = self.stft_discriminator(mel_x)
            d_stft_loss_real = self.disc_loss_fn(d_output_real_stft, torch.ones_like(d_output_real_stft))
            d_stft_loss_fake = self.disc_loss_fn(d_output_fake_stft, torch.zeros_like(d_output_fake_stft))
            d_stft_loss = (d_stft_loss_real + d_stft_loss_fake) / 2

        self.log_step_metrics(recon_loss, g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss, embedding_loss, perplexity, 0, 0, x_hat, x, z_q, y_true, batch_idx, step_name='val')

    # @torch.no_grad()
    # @skip_if_sanity_checking
    # def log_step_metrics(self, recon_loss, g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss, embedding_loss, perplexity, x_hat, x, batch_idx, step_name):
    #     super().log_step_metrics(recon_loss, g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss, embedding_loss, perplexity, x_hat, x, batch_idx, step_name)

    #     for emotion in range(8):
    #         self.decoder(x, z_q, label)

    @torch.no_grad()
    @skip_if_sanity_checking
    def log_step_metrics(self, recon_loss, g_wave_loss, g_stft_loss, d_wave_loss, d_stft_loss, x_hat, x, z_q, label, batch_idx, step_name):
        # log loss
        self.log(f'{step_name}_g_recons_loss', recon_loss, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{step_name}_g_wave_loss', g_wave_loss, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{step_name}_g_stft_loss', g_stft_loss, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{step_name}_d_wave_loss', d_wave_loss, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{step_name}_d_stft_loss', d_stft_loss, sync_dist=True, batch_size=self.batch_size)

        # log image metrics
        self.log(f'{step_name}_ssim', self.ssim(self.mel_transform(x_hat), self.mel_transform(x)), sync_dist=True, batch_size=self.batch_size)

        # log audio metrics every 10th batch, since they're calculated on CPU and it takes some time
        if batch_idx % 10 == 0:
            for metric_name, metric_func in [(f'{step_name}_pesq', self.pesq), (f'{step_name}_stoi', self.stoi)]:
                try:
                    self.log(metric_name, metric_func(x_hat, x), sync_dist=True, batch_size=self.batch_size)
                except TypeError:
                    continue

        # log exemplary data, let's save 5 examples per epoch
        if batch_idx < 5:
            self.logger.experiment.log_audio(audio_data=x[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'{step_name}_epoch_{self.trainer.current_epoch}_{batch_idx}.wav')
            self.logger.experiment.log_audio(audio_data=x_hat[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'{step_name}_epoch_{self.trainer.current_epoch}_{batch_idx}_recons.wav')
            self.logger.experiment.log_image(image_data=self.mel_transform(x[0]).to('cpu').numpy(), image_channels='first', name=f'{step_name}_epoch_{self.trainer.current_epoch}_{batch_idx}_target')
            self.logger.experiment.log_image(image_data=self.mel_transform(x_hat[0]).to('cpu').numpy(), image_channels='first', name=f'{step_name}_epoch_{self.trainer.current_epoch}_{batch_idx}_pred')

            for emotion in range(8):
                x_hat, *_ = self.decoder(x, z_q, emotion)
                self.logger.experiment.log_audio(audio_data=x_hat[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'{step_name}_epoch_{self.trainer.current_epoch}_{batch_idx}_emotion{emotion}.wav')

    @rank_zero_only  # needed for self.logger.experiment.get_key() to work properly when using >1 GPUs
    def log_best_checkpoint(self):
        best_checkpoint = os_sorted(os.listdir(f'{self.results_root}/{self.logger.experiment.get_key()}/checkpoints'))[-1]
        self.logger.experiment.log_asset(f'{self.results_root}/{self.logger.experiment.get_key()}/checkpoints/{best_checkpoint}')
        create_marker_file(f'{self.results_root}/{self.logger.experiment.get_key()}')




class DualLatentWithSwappingAE(VQVAE):
    def __init__(self, args_dict):
        super(DualLatentWithSwappingAE, self).__init__(args_dict)
        del self.vector_quantization

        self.encoder = DualLatentEncoder(in_dim=self.in_dim,
                                         h_dim=self.h_dim,
                                         cont_latent_dim=self.cont_latent_dim, 
                                         vq_latent_dim=self.vq_latent_dim,
                                         n_e=self.n_e,
                                         beta=self.beta)

        self.decoder = DualLatentDecoder(in_dim=self.latent_dim, h_dim=self.h_dim) # Generalnie musi być tak, że latent_dim = cont_latent_dim + vq_latent_dim
        
        self.patch_encoder = PatchEncoder()
        self.cooc_discriminator = CooccurencePatchDiscriminator()
        
    def configure_optimizers(self):
        self.encoder.vector_quantization[1].set_device(self.device)
        for transform in self.mel_transform.transforms:
            if isinstance(transform, torch.nn.Module):
                transform.to(self.device)

        optimizer_g = torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.g_learning_rate, weight_decay=self.weight_decay)
        optimizer_d = torch.optim.Adam(chain(self.wave_discriminator.parameters()), lr=self.d_learning_rate, weight_decay=self.weight_decay)
        optimizer_cooc = torch.optim.Adam(chain(self.patch_encoder.parameters(), self.cooc_discriminator.parameters()), lr=self.d_learning_rate, weight_decay=self.weight_decay)
        return [optimizer_g, optimizer_d, optimizer_cooc], []

    def crop_patches(self, image, num_patches=8):
        patches = []
        for _ in range(num_patches):
            patches.append(crop_random_patch(image))
        return torch.stack(patches)

    def forward(self, x12):
        x1 = x12[:, 0]
        x2 = x12[:, 1]
        encoded_x1, z_q1, embedding_loss1, perplexity1 = self.encoder(x1)
        encoded_x2, z_q2, embedding_loss2, perplexity2 = self.encoder(x2)
        
        x_hat1 = self.decoder(encoded_x1, z_q1)
        x_hat2 = self.decoder(encoded_x2, z_q1)
        
        return x_hat1, x_hat2, embedding_loss1, embedding_loss2, perplexity1, perplexity2

    def training_step(self, batch, batch_idx):
        if batch.shape[0] % 2 != 0:
            batch = batch[:-1]
        batch = batch.view(batch.shape[0] // 2, 2, 1, batch.shape[-1]).to(self.device)
        optimizer_g, optimizer_d, optimizer_cooc = self.optimizers()
        
        x_hat1, x_hat2, embedding_loss1, embedding_loss2, perplexity1, perplexity2 = self(batch)
        embedding_loss1 *= 100
        embedding_loss2 *= 100

        # Train generator
        self.toggle_optimizer(optimizer_g)
        recon_loss = self.loss_fn(x_hat1, batch[:, 0]) * 100
        
        g_wave_reconstruction_fake = self.wave_discriminator(x_hat1)
        g_wave_swapping_fake = self.wave_discriminator(x_hat2)
        g_wave_reconstruction_loss = self.disc_loss_fn(g_wave_reconstruction_fake, torch.ones_like(g_wave_reconstruction_fake))
        g_wave_swapping_loss = self.disc_loss_fn(g_wave_swapping_fake, torch.ones_like(g_wave_swapping_fake))
        g_wave_loss = (g_wave_reconstruction_loss + g_wave_swapping_loss) / 2
        g_loss = (recon_loss + embedding_loss1 + embedding_loss2) + g_wave_loss

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # Train discriminators
        self.toggle_optimizer(optimizer_cooc)

        x_hat1, x_hat2, *_ = self(batch)

        spectrograms_real = self.mel_transform(batch[:, 1]) # TU MA BYĆ 1 BO STYL
        spectrograms_swapped = self.mel_transform(x_hat2)

        target_patches_raw = self.crop_patches(spectrograms_real)
        target_patches = target_patches_raw.view(-1, 1, 32, 32)
        target_features = self.patch_encoder(target_patches)
        target_features = torch.flatten(target_features, start_dim=1)

        mix_patches = self.crop_patches(spectrograms_swapped)
        mix_patches = mix_patches.view(-1, 1, 32, 32)
        mix_features = self.patch_encoder(mix_patches)
        mix_features = torch.flatten(mix_features, start_dim=1)

        ref_patch = torch.mean(target_patches_raw, dim=0)
        ref_patch = ref_patch.unsqueeze(1).repeat(1, 8, 1, 1, 1).reshape(-1, 1, 32, 32)
        ref_features = self.patch_encoder(ref_patch)
        ref_features = torch.flatten(ref_features, start_dim=1)
        
        cooc_output_target = self.cooc_discriminator(ref_features, target_features)
        cooc_loss_target = self.disc_loss_fn(cooc_output_target, torch.ones_like(cooc_output_target))

        cooc_output_mix = self.cooc_discriminator(ref_features, mix_features)
        cooc_loss_mix = self.disc_loss_fn(cooc_output_mix, torch.zeros_like(cooc_output_mix))

        cooc_loss_d = (cooc_loss_target + cooc_loss_mix) / 2

        self.manual_backward(cooc_loss_d)
        optimizer_cooc.step()
        optimizer_cooc.zero_grad()
        self.untoggle_optimizer(optimizer_cooc)

        self.toggle_optimizer(optimizer_d)

        d_output_reconstruction_real = self.wave_discriminator(batch[:, 0])
        d_output_reconstruction_fake = self.wave_discriminator(x_hat1)
        d_output_swapp_real = self.wave_discriminator(batch[:, 1])
        d_output_swapp_fake = self.wave_discriminator(x_hat2)

        d_reconstruction_loss_real = self.disc_loss_fn(d_output_reconstruction_real, torch.ones_like(d_output_reconstruction_real))
        d_reconstruction_loss_fake = self.disc_loss_fn(d_output_reconstruction_fake, torch.zeros_like(d_output_reconstruction_fake))
        d_swapp_loss_real = self.disc_loss_fn(d_output_swapp_real, torch.ones_like(d_output_swapp_real))
        d_swapp_loss_fake = self.disc_loss_fn(d_output_swapp_fake, torch.zeros_like(d_output_swapp_fake))

        d_reconstruction_loss = (d_reconstruction_loss_real + d_reconstruction_loss_fake) / 2
        d_swap_loss = (d_swapp_loss_real + d_swapp_loss_fake) / 2
        
        d_result_loss = d_reconstruction_loss + d_swap_loss

        self.manual_backward(d_result_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # Log metrics
        self.log_training_step_metrics(recon_loss, g_wave_loss, d_result_loss, embedding_loss1, embedding_loss2, perplexity1, perplexity2, cooc_loss_d, target_patches, mix_patches, ref_patch, x_hat1, x_hat2, batch, batch_idx)


    def validation_step(self, batch, batch_idx):
        if batch.shape[0] % 2 != 0:
            batch = batch[:-1]
        batch = batch.view(batch.shape[0] // 2, 2, 1, batch.shape[-1]).to(self.device)
        x_hat1, x_hat2, embedding_loss1, embedding_loss2, perplexity1, perplexity2 = self(batch)
        recon_loss = self.loss_fn(x_hat1, batch[:,0])

        # Wave discriminator calculations
        g_output_recenstruction_fake = self.wave_discriminator(x_hat1)
        g_wave_reconstruction_loss = self.disc_loss_fn(g_output_recenstruction_fake, torch.ones_like(g_output_recenstruction_fake))

        g_output_swapp_fake = self.wave_discriminator(x_hat2)
        g_wave_swapp_loss = self.disc_loss_fn(g_output_swapp_fake, torch.ones_like(g_output_swapp_fake))
        
        g_wave_loss = (g_wave_reconstruction_loss + g_wave_swapp_loss) / 2

        # Discriminator loss
        d_output_reconstruction_real = self.wave_discriminator(batch[:,0])
        d_output_reconstruction_fake = self.wave_discriminator(x_hat1)
        
        d_output_swapp_real = self.wave_discriminator(batch[:,1])
        d_output_swapp_fake = self.wave_discriminator(x_hat2)

        d_wave_reconstruction_loss_real = self.disc_loss_fn(d_output_reconstruction_real, torch.ones_like(d_output_reconstruction_real))
        d_wave_reconstruction_loss_fake = self.disc_loss_fn(d_output_reconstruction_fake, torch.zeros_like(d_output_reconstruction_fake))
        
        d_wave_swapp_loss_real = self.disc_loss_fn(d_output_swapp_real, torch.ones_like(d_output_swapp_real))
        d_wave_swapp_loss_fake = self.disc_loss_fn(d_output_swapp_fake, torch.zeros_like(d_output_swapp_fake))

        d_wave_reconstruction_loss = (d_wave_reconstruction_loss_real + d_wave_reconstruction_loss_fake) / 2
        d_wave_swapp_loss = (d_wave_swapp_loss_real + d_wave_swapp_loss_fake) / 2
        
        d_wave_loss = (d_wave_reconstruction_loss + d_wave_swapp_loss) / 2

        # Co-occurrence discriminator calculations
        spectrograms_real = self.mel_transform(batch[:, 0])
        spectrograms_swapped = self.mel_transform(x_hat2)

        target_patches_raw = self.crop_patches(spectrograms_real)
        target_patches = target_patches_raw.view(-1, 1, 32, 32)
        target_features = self.patch_encoder(target_patches)
        target_features = torch.flatten(target_features, start_dim=1)

        mix_patches = self.crop_patches(spectrograms_swapped)
        mix_patches = mix_patches.view(-1, 1, 32, 32)
        mix_features = self.patch_encoder(mix_patches)
        mix_features = torch.flatten(mix_features, start_dim=1)

        ref_patch = torch.mean(target_patches_raw, dim=0)
        ref_patch = ref_patch.unsqueeze(1).repeat(1, 8, 1, 1, 1).reshape(-1, 1, 32, 32)
        ref_features = self.patch_encoder(ref_patch)
        ref_features = torch.flatten(ref_features, start_dim=1)
        
        cooc_output_target = self.cooc_discriminator(ref_features, target_features)
        cooc_loss_target = self.disc_loss_fn(cooc_output_target, torch.ones_like(cooc_output_target))

        cooc_output_mix = self.cooc_discriminator(ref_features, mix_features)
        cooc_loss_mix = self.disc_loss_fn(cooc_output_mix, torch.zeros_like(cooc_output_mix))

        cooc_loss_d = (cooc_loss_target + cooc_loss_mix) / 2

        self.log_validation_step_metrics(recon_loss, g_wave_loss, d_wave_loss, embedding_loss1, embedding_loss2, perplexity1, perplexity2, cooc_loss_d, target_patches, mix_patches, ref_patch, x_hat1, x_hat2, batch, batch_idx)


    @torch.no_grad()
    def log_training_step_metrics(self, recon_loss, g_wave_loss, d_wave_loss, embedding_loss1, embedding_loss2, perplexity1, perplexity2, cooc_loss_d, target_patches, mix_patches, ref_patch, x_hat1, x_hat2, batch, batch_idx):
        # log loss
        self.log('train_g_recons_loss', recon_loss, sync_dist=True, batch_size=self.batch_size)
        self.log('train_g_wave_loss', g_wave_loss, sync_dist=True, batch_size=self.batch_size)
        self.log('train_d_wave_loss', d_wave_loss, sync_dist=True, batch_size=self.batch_size)
        self.log('train_d_cooc_loss', cooc_loss_d, sync_dist=True, batch_size=self.batch_size)

        # log image metrics
        self.log('train_reconstruction_ssim', self.ssim(self.mel_transform(x_hat1), self.mel_transform(batch[:,0])), sync_dist=True, batch_size=self.batch_size)
        self.log('train_swap_ssim', self.ssim(self.mel_transform(x_hat2), self.mel_transform(batch[:,1])), sync_dist=True, batch_size=self.batch_size)

        # log audio metrics every 10th batch, since they're calculated on CPU and it takes some time
        if batch_idx % 10 == 0:
            for metric_name, metric_func in [('train_pesq', self.pesq), ('train_stoi', self.stoi)]:
                try:
                    self.log(metric_name+"reconstruction", metric_func(x_hat1, batch[:,0]), sync_dist=True, batch_size=self.batch_size)
                    self.log(metric_name+"swapp", metric_func(x_hat2, batch[:, 1]), sync_dist=True, batch_size=self.batch_size)
                except TypeError:
                    continue


        # log exemplary data, let's save 5 examples per epoch
        if batch_idx < 5:
            self.logger.experiment.log_audio(audio_data=batch[0][0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_content.wav')
            self.logger.experiment.log_audio(audio_data=batch[0][1][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_style.wav')

            self.logger.experiment.log_audio(audio_data=x_hat1[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_recons.wav')
            self.logger.experiment.log_audio(audio_data=x_hat2[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_swapped.wav')
            
            self.logger.experiment.log_image(image_data=self.mel_transform(batch[0][0]).to('cpu').numpy(), image_channels='first', name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_target_content')
            self.logger.experiment.log_image(image_data=self.mel_transform(batch[0][1]).to('cpu').numpy(), image_channels='first', name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_target_style')

            self.logger.experiment.log_image(image_data=self.mel_transform(x_hat1[0]).to('cpu').numpy(), image_channels='first', name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_reconstructed')
            self.logger.experiment.log_image(image_data=self.mel_transform(x_hat2[0]).to('cpu').numpy(), image_channels='first', name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_swapped')

            # # Log the patches used for cooc discriminator
            # target_patches_first8 = target_patches[:8]
            # mix_patches_first8 = mix_patches[:8]
            # ref_patch_first8 = ref_patch[:8]

            # # Create grids of images for better visualization
            # target_grid = torchvision.utils.make_grid(target_patches_first8, nrow=4, normalize=True, scale_each=True)
            # mix_grid = torchvision.utils.make_grid(mix_patches_first8, nrow=4, normalize=True, scale_each=True)
            # ref_grid = torchvision.utils.make_grid(ref_patch_first8, nrow=4, normalize=True, scale_each=True)

            # # Log images
            # self.logger.experiment.log_image(image_data=target_grid.cpu().numpy(), image_channels='first', name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_target_patch')
            # self.logger.experiment.log_image(image_data=mix_grid.cpu().numpy(), image_channels='first', name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_swapped_patch')
            # self.logger.experiment.log_image(image_data=ref_grid.cpu().numpy(), image_channels='first', name=f'train_epoch_{self.trainer.current_epoch}_{batch_idx}_ref_patch')

        # Log additional metrics specific to VQVAE
        self.log('train_reconstruction_embedding_loss', embedding_loss1, sync_dist=True, batch_size=self.batch_size)
        self.log('train_swapp_embedding_loss', embedding_loss2, sync_dist=True, batch_size=self.batch_size)
        self.log('train_reconstruction_perplexity', perplexity1, sync_dist=True, batch_size=self.batch_size)
        self.log('train_swapp_perplexity', perplexity2, sync_dist=True, batch_size=self.batch_size)

    @torch.no_grad()
    @skip_if_sanity_checking
    def log_validation_step_metrics(self, recon_loss, g_wave_loss, d_wave_loss, embedding_loss1, embedding_loss2, perplexity1, perplexity2, cooc_loss_d, target_patches, mix_patches, ref_patch, x_hat1, x_hat2, batch, batch_idx):
        self.log('val_g_recons_loss', recon_loss, sync_dist=True, batch_size=self.batch_size)
        self.log('val_g_wave_loss', g_wave_loss, sync_dist=True, batch_size=self.batch_size)
        self.log('val_d_wave_loss', d_wave_loss, sync_dist=True, batch_size=self.batch_size)
        self.log('val_d_cooc_loss', cooc_loss_d, sync_dist=True, batch_size=self.batch_size)

        # log image metrics
        self.log('val_reconstruction_ssim', self.ssim(self.mel_transform(x_hat1), self.mel_transform(batch[:,0])), sync_dist=True, batch_size=self.batch_size)
        self.log('val_swapp_ssim', self.ssim(self.mel_transform(x_hat2), self.mel_transform(batch[:,1])), sync_dist=True, batch_size=self.batch_size)

        # log audio metrics every 10th batch, since they're calculated on CPU and it takes some time
        if batch_idx % 10 == 0:
            for metric_name, metric_func in [('val_pesq', self.pesq), ('val_stoi', self.stoi)]:
                try:
                    self.log(metric_name+"reconstruction", metric_func(x_hat1, batch[:,0]), sync_dist=True, batch_size=self.batch_size)
                    self.log(metric_name+"swapp", metric_func(x_hat2, batch[:, 1]), sync_dist=True, batch_size=self.batch_size)
                except TypeError:
                    continue

        # log exemplary data, let's save 5 examples per epoch
        if batch_idx < 5:
            self.logger.experiment.log_audio(audio_data=batch[0][0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_content.wav')
            self.logger.experiment.log_audio(audio_data=batch[0][1][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_style.wav')
            
            self.logger.experiment.log_audio(audio_data=x_hat1[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_reconstruction.wav')
            self.logger.experiment.log_audio(audio_data=x_hat2[0][0].to('cpu').numpy().astype(np.float32), sample_rate=self.sr, file_name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_swapp.wav')
            
            self.logger.experiment.log_image(image_data=self.mel_transform(batch[0][0]).to('cpu').numpy(), image_channels='first', name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_content')
            self.logger.experiment.log_image(image_data=self.mel_transform(batch[0][1]).to('cpu').numpy(), image_channels='first', name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_style')
            self.logger.experiment.log_image(image_data=self.mel_transform(x_hat1[0]).to('cpu').numpy(), image_channels='first', name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_reconstruction')
            self.logger.experiment.log_image(image_data=self.mel_transform(x_hat2[0]).to('cpu').numpy(), image_channels='first', name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_swapp')

            # # Log the patches used for cooc discriminator
            # target_patches_first8 = target_patches[:8]
            # mix_patches_first8 = mix_patches[:8]
            # ref_patch_first8 = ref_patch[:8]

            # # Create grids of images for better visualization
            # target_grid = torchvision.utils.make_grid(target_patches_first8, nrow=4, normalize=True, scale_each=True)
            # mix_grid = torchvision.utils.make_grid(mix_patches_first8, nrow=4, normalize=True, scale_each=True)
            # ref_grid = torchvision.utils.make_grid(ref_patch_first8, nrow=4, normalize=True, scale_each=True)

            # # Log images
            # self.logger.experiment.log_image(image_data=target_grid.cpu().numpy(), image_channels='first', name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_target_patch')
            # self.logger.experiment.log_image(image_data=mix_grid.cpu().numpy(), image_channels='first', name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_swapped_patch')
            # self.logger.experiment.log_image(image_data=ref_grid.cpu().numpy(), image_channels='first', name=f'val_epoch_{self.trainer.current_epoch}_{batch_idx}_ref_patch')

        # Log additional metrics specific to VQVAE
        self.log('val_reconstruction_embedding_loss', embedding_loss1, sync_dist=True, batch_size=self.batch_size)
        self.log('val_swapp_embedding_loss', embedding_loss2, sync_dist=True, batch_size=self.batch_size)
        self.log('val_reconstruction_perplexity', perplexity1, sync_dist=True, batch_size=self.batch_size)
        self.log('val_swapp_perplexity', perplexity2, sync_dist=True, batch_size=self.batch_size)