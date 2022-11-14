import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from shutil import rmtree

import torch
import optuna
import argparse

from omegaconf import OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from termcolor import colored

from dataset.synth_datamodule import ModularSynthDataModule
from utils.gpu_utils import get_device

from model.lit_module import LitModularSynth
from model.lit_module_decoder_only import LitModularSynthDecOnly
from utils.train_utils import get_project_root


root = get_project_root()
EXP_ROOT = root.joinpath('experiments', 'current')
DATA_ROOT = root.joinpath('data')


# todo: consider delete
class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics['train_losses/train_lsd'])

def objective(trial: optuna.trial.Trial, run_args) -> float:

    exp_name = run_args.experiment
    dataset_name = run_args.dataset

    cfg = configure_experiment(exp_name, dataset_name, run_args.config, run_args.debug)

    lr = trial.suggest_float("lr", 1e-6, 1e-2)
    cfg.model.optimizer.base_lr = lr

    datamodule = ModularSynthDataModule(cfg.data_dir, cfg.model.batch_size, cfg.model.num_workers,
                                        added_noise_std=cfg.synth.added_noise_std)

    # todo: allow config of out of domain data
    datamodule.setup()

    device = get_device(run_args.gpu_index)

    lit_module = LitModularSynthDecOnly(cfg, device, run_args, tuning_mode=True)
    if cfg.model.get('ckpt_path', None):
        lit_module.load_from_checkpoint(checkpoint_path=cfg.model.ckpt_path, train_cfg=cfg, device=device)
    lsd_metrics = MetricsCallback()
    callbacks = [LearningRateMonitor(logging_interval='step'),
                 MetricsCallback(),
                 PyTorchLightningPruningCallback(trial, monitor="train_lsd_val")]

    tb_logger = TensorBoardLogger(cfg.logs_dir, name=exp_name)
    lit_module.tb_logger = tb_logger.experiment

    if len(datamodule.train_dataset.params) < 50:
        log_every_n_steps = len(datamodule.train_dataset.params)
    else:
        log_every_n_steps = 50

    trainer = Trainer(logger=tb_logger,
                      callbacks=callbacks,
                      max_epochs=cfg.model.num_epochs,
                      auto_select_gpus=True,
                      devices=[run_args.gpu_index],
                      accelerator="gpu",
                      detect_anomaly=True,
                      log_every_n_steps=log_every_n_steps,
                      check_val_every_n_epoch=500,
                      enable_checkpointing=False)

    hyperparameters = dict(lr=cfg.model.optimizer.base_lr)
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(lit_module, datamodule=datamodule)
    lsd = trainer.callback_metrics['train_losses/train_lsd']

    return lsd


def configure_experiment(exp_name: str, dataset_name: str, config_name: str, debug: bool = False):

    exp_dir = os.path.join(EXP_ROOT, exp_name, '')
    data_dir = os.path.join(DATA_ROOT, dataset_name, '')
    config_path = os.path.join(root, 'configs', config_name)

    if os.path.isdir(exp_dir):
        if not debug:
            overwrite = input(colored(f"Folder {exp_dir} already exists. Overwrite previous experiment (Y/N)?"
                                      f"\n\tThis will delete all files related to the previous run!",
                                      'yellow'))
            if overwrite.lower() != 'y':
                print('Exiting...')
                exit()

        print("Deleting previous experiment...")
        rmtree(exp_dir)

    cfg = OmegaConf.load(config_path)

    cfg.exp_dir = exp_dir
    cfg.data_dir = data_dir
    cfg.ckpts_dir = os.path.join(exp_dir, 'checkpoints', '')
    cfg.logs_dir = os.path.join(exp_dir, 'tensorboard', '')

    config_dump_dir = os.path.join(cfg.exp_dir, 'config_dump', '')
    os.makedirs(config_dump_dir, exist_ok=True)

    config_dump_path = os.path.join(config_dump_dir, 'config.yaml')
    OmegaConf.save(cfg, config_dump_path)

    return cfg


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Train AI Synth')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing) -1 for cpu',
                        type=int, default=0)
    parser.add_argument('-e', '--experiment', required=True,
                        help='Experiment name', type=str)
    parser.add_argument('-d', '--dataset', required=True, type=str,
                        help='Dataset name')
    parser.add_argument('-c', '--config', required=True, type=str,
                        help='configuration file path')
    parser.add_argument('-de', '--debug', required=False, action='store_true',
                        help='run in debug mode', default=False)
    parser.add_argument('-p', '--pruning', required=False, action='store_true',
                        help='use pruning in optuna', default=False)

    args = parser.parse_args()

    args.params_to_freeze = {(0, 0): ['freq'], (0, 1): ['waveform', 'mod_index']}

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(lambda x: objective(x, args), n_trials=3, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    for trial in study.trials:
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
