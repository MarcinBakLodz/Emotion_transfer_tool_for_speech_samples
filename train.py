import comet_ml  # this needs to be imported before torch, that's how comet_ml works for some reason
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
import torch
from torchinfo import summary

from datasets.ravdess import RAVDESS
from datasets.vctk import VCTK
from models.ae import AE, VQVAE, DualLatentAE
from utils import get_parser_from_json


def get_dset(train_share=0.8):
    # dset = VCTK(root_dir='../data')
    dset = RAVDESS(root_dir='../data/RAVDESS')
    train_size = int(train_share * len(dset))
    test_size = len(dset) - train_size
    return torch.utils.data.random_split(dataset=dset, lengths=[train_size, test_size], generator=torch.Generator().manual_seed(42))  # fix the generator for reproducible results


def set_up_comet_logger(model, model_config, test_sample, tags):
    comet_logger = CometLogger()  # https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/
    comet_logger.log_hyperparams(vars(model_config))

    for tag in tags:
        comet_logger.experiment.add_tag(tag)

    # log code
    for file in [f for f in os.listdir(os.path.curdir) if (f.endswith('.py') or f.endswith('.yml'))]:
        comet_logger.experiment.log_code(file_name=file)
    for file in [f for f in os.listdir('datasets') if f.endswith('.py')]:
        comet_logger.experiment.log_code(file_name=os.path.join('datasets', file))
    for file in [f for f in os.listdir('models') if (f.endswith('.py') or f.endswith('.json'))]:
        comet_logger.experiment.log_code(file_name=os.path.join('models', file))

    # log number of parameters
    total_params = 0
    for _, para in model.named_parameters():
        total_params += torch.numel(para.data)
    comet_logger.experiment.log_parameter(name="n_params", value=total_params)

    # log summary
    summ = summary(model=model, input_data=test_sample.to(next(model.parameters()).device), device=next(model.parameters()).device, verbose=0)
    comet_logger.experiment.set_model_graph(graph=f"{model.__repr__()}\n\n{summ}")

    return comet_logger


def set_up_callbacks(experiment_key, es_min_delta=1e-9, es_patience=100, chckpt_save_top_k=5):
    early_stop_callback = EarlyStopping(monitor='val_g_recons_loss', min_delta=es_min_delta, patience=es_patience, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_g_recons_loss', dirpath=f'../results/{experiment_key}/checkpoints', filename='{epoch:02d}-{val_loss:.2f}', save_top_k=chckpt_save_top_k, mode='min')
    return [early_stop_callback, checkpoint_callback]


def training():
    model_config = get_parser_from_json('models/dual_latent_ae_config.json')
    model = DualLatentAE(args_dict=vars(model_config))

    train_dataset, test_dataset = get_dset()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=model_config.batch_size, shuffle=True, pin_memory=True, num_workers=os.cpu_count())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=model_config.batch_size, drop_last=True, pin_memory=True, num_workers=os.cpu_count())

    comet_logger = set_up_comet_logger(model=model, model_config=model_config, test_sample=next(iter(test_loader)), tags=[model_config.name, 'RAVDESS', 'LeakyReLU', 'NEAREST', 'WAVE DISCRIMINATOR'])

    trainer = Trainer(callbacks=set_up_callbacks(comet_logger.experiment.get_key()),  # https://lightning.ai/docs/pytorch/stable/common/trainer.html#
                      logger=comet_logger,
                      log_every_n_steps=10,
                      accelerator='auto',
                      devices='auto',
                      precision='32-true',
                      max_epochs=1000)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    training()
