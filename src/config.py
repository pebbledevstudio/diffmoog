import json
import os
from json import dump
from dataclasses import dataclass, field, asdict
from pathlib import Path, WindowsPath
from shutil import rmtree
from termcolor import colored

import numpy as np

# from synth_config import BASIC_FLOW
from typing import Dict, List

from torch.utils.tensorboard import SummaryWriter

from synth.synth_constants import synth_structure

# root = r'/home/almogelharar/almog/ai_synth/'
root = r'C:\Users\Noy Uzrad\PycharmProjects\ai_synth'
EXP_ROOT = os.path.join(root, 'experiments')
DATA_ROOT = os.path.join(root, 'data')


@dataclass
class Config:

    " Mode - define a common configuration for the whole system     "
    "   0 -                     Use custom configurations           "
    "   Any other number -      Use predefined configuration preset. See below "
    mode: int = 1

    " The architecture of the system, that defines the data flow and the loss functions:                    "
    "   1. SPECTROGRAM_ONLY (input -> CNN -> parameters -> Synth -> output; Loss over spectrograms)         "
    "   2. PARAMETERS_ONLY (input -> CNN -> parameters; Loss over parameters)                               "
    "   3. FULL - (input -> CNN -> parameters -> Synth -> output; Loss over spectrograms AND parameters)    "
    "   4. SPEC_NO_SYNTH (input -> CNN -> parameters); Output inner product <probabilities, spectrograms>;   "
    "      Loss over spectrograms)                                                                          "
    "   5. REINFORCE - (input -> CNN -> parameters); Loss is computed to maximize rewards for correct       "
    "       classification. Using the classical REINFORCE algorithm                                         "
    architecture: str = 'SPECTROGRAM_ONLY'  # SPECTROGRAM_ONLY, PARAMETERS_ONLY, SPEC_NO_SYNTH or FULL (Spectrogram + parameters)

    " Spectrogram loss type options:" \
    "1. MSE" \
    "2. LSD (Log Spectral Distance)" \
    "3. KL (Kullback-Leibler)" \
    "4. EMD (earth movers distance)" \
    "5. MULTI-SPECTRAL"
    spectrogram_loss_type: str = 'MULTI-SPECTRAL'
    freq_param_loss_type: str = 'MSE'  # MSE or CE (Cross Entropy)

    " The model can output the oscillator frequency as:                                 "
    "   1. LOGITS (size is num of frequencies, for cross entropy loss)                  "
    "   2. PROBS (same as LOGITS, but softmax is applied)                               "
    "   3. WEIGHTED - inner product of <probabilities, original frequencies>. size is 1 "
    "   4. SINGLE - linear layer outputs single neuron. size is 1                       "
    model_frequency_output: str = 'SINGLE'
    transform: str = 'MEL_SPECTROGRAM'  # MEL_SPECTROGRAM or SPECTROGRAM- to be used in the data loader and at the synth output

    use_loaded_model = False

    project_root: str = None
    tensorboard_logdir: str = None
    ckpts_dir: str = None
    artifacts_dir: str = None

    save_model_path: str = None
    # load_model_path = Path(__file__).parent.parent.joinpath('trained_models', 'trained_synth_net.pt')
    load_model_path: str = None

    txt_path: str = None
    numpy_path: str = None

    num_epochs_to_save_model: int = 10

    regression_loss_factor: float = 1e-1
    spectrogram_loss_factor: float = 1e-5
    freq_mse_loss_factor: float = 1e-3
    freq_reinforce_loss_factor: float = 1e5

    multi_spectral_loss_spec_type: str = 'BOTH'
    multi_spectral_loss_preset: str = 'cumsum_time'

    add_parameters_loss = True
    parameters_loss_type = 'L1'
    parameters_loss_weight = 1 / 100
    spectrogram_loss_weight = 1 / 100

    spectrogram_loss_warmup = 40 * 1000
    loss_switch_steps = 40 * 1000
    min_parameters_loss_decay = 0.01

    use_chain_loss = True


@dataclass
class ModelConfig:
    preset: str = 'MODULAR'
    model_type: str = 'simple'
    backbone: str = 'resnet'
    batch_size: int = 50
    num_epochs: int = 120
    learning_rate: float = 3e-4
    optimizer_weight_decay: float = 0
    num_workers: int = 0
