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
root = r'/home/almogelharar/almog/ai_synth/'
EXP_ROOT = os.path.join(root, 'experiments')
DATA_ROOT = os.path.join(root, 'data')


@dataclass
class Config:

    sample_rate: int = 16000
    signal_duration_sec: float = 4.0

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
    multi_spectral_loss_preset: str = 'mag_logmag'

    add_parameters_loss = True
    parameters_loss_type = 'L1'
    parameters_loss_weight = 1 / 100
    spectrogram_loss_weight = 1 / 50000

    spectrogram_loss_warmup = 40 * 100
    loss_switch_steps = 20 * 100

    smoothness_loss_weight = 0

    # Debug
    debug_mode: bool = False
    plot_spec: bool = False
    print_train_batch_stats: bool = False
    print_timings: bool = False
    print_synth_param_stats: bool = False
    print_accuracy_stats: bool = False
    print_per_accuracy_stats_multiple_epochs: bool = True

    log_spectrogram_mse_loss: bool = False

    def __init__(self, project_root: str = ''):

        if project_root == '':
            return

        self.project_root = project_root

        self.tensorboard_logdir = os.path.join(project_root, 'tensorboard', '')
        self.ckpts_dir = os.path.join(project_root, 'ckpts', '')
        os.makedirs(self.ckpts_dir, exist_ok=True)

        self.save_model_path = os.path.join(project_root, 'ckpts', 'trained_synth_net.pt')
        # load_model_path = Path(__file__).parent.parent.joinpath('trained_models', 'trained_synth_net.pt')
        self.load_model_path = '' #'/home/almogelharar/almog/ai_synth/experiments/current/basic_flow_test/ckpts/synth_net_epoch10.pt'

        self.artifacts_dir = os.path.join(project_root, 'artifacts', '')
        os.makedirs(self.artifacts_dir, exist_ok=True)

        self.txt_path = os.path.join(project_root, 'artifacts', 'loss_list.txt')
        self.numpy_path = os.path.join(project_root, 'artifacts', 'loss_list.npy')

        self.__post_init__()

    def __post_init__(self):

        if self.log_spectrogram_mse_loss:
            self.spectrogram_loss_factor = 1000

        if self.freq_param_loss_type == 'CE':
            self.model_frequency_output = 'LOGITS'

        if self.spectrogram_loss_type == 'MULTI-SPECTRAL':
            # one of ['BOTH', 'MEL_SPECTROGRAM', 'SPECTROGRAM']
            self.multi_spectral_loss_spec_type = 'SPECTROGRAM'

        if self.mode == 1:
            self.architecture = 'SPECTROGRAM_ONLY'
            self.spectrogram_loss_type = 'MULTI-SPECTRAL'
            self.model_frequency_output = 'SINGLE'


@dataclass
class DatasetConfig:
    batch_size: int = 10
    num_epochs_to_print_stats: int = 100
    train_parameters_file: str = None
    train_audio_dir: str = None
    val_parameters_file: str = None
    val_audio_dir: str = None
    test_parameters_file: str = None
    test_audio_dir: str = None
    inference_audio_dir: str = None
    inference_plots_dir: str = None
    train_dataset_dir_path: str = None
    test_dataset_dir_path: str = None
    val_dataset_dir_path: str = None

    def __init__(self, dataset_name):

        dataset_dir = os.path.join(DATA_ROOT, dataset_name, '')

        self.train_parameters_file: str = os.path.join(dataset_dir, 'train', 'params_dataset.pkl')
        self.train_audio_dir: str = os.path.join(dataset_dir, 'train', 'wav_files')
        self.val_parameters_file: str = os.path.join(dataset_dir, 'val', 'params_dataset.pkl')
        self.val_audio_dir: str = os.path.join(dataset_dir, 'val', 'wav_files')
        self.test_parameters_file: str = os.path.join(dataset_dir, 'test', 'params_dataset.pkl')
        self.test_audio_dir: str = os.path.join(dataset_dir, 'test', 'wav_files')
        self.inference_audio_dir: str = os.path.join(dataset_dir, 'test', 'inference_wav_files')
        self.inference_plots_dir: str = os.path.join(dataset_dir, 'test', 'inference_plots')
        self.train_dataset_dir_path: str = os.path.join(dataset_dir, 'train')
        self.test_dataset_dir_path: str = os.path.join(dataset_dir, 'test')
        self.val_dataset_dir_path: str = os.path.join(dataset_dir, 'val')


@dataclass
class ModelConfig:
    preset: str = 'MODULAR'
    model_type: str = 'simple'
    backbone: str = 'resnet'
    batch_size: int = 2
    num_epochs: int = 100
    learning_rate: float = 3e-4
    optimizer_weight_decay: float = 0
    optimizer_scheduler_lr: float = 0
    optimizer_scheduler_gamma: float = 0.1
    reinforcement_epsilon: float = 0.15
    num_workers: int = 2


@dataclass
class SynthConfig:
    preset: str = 'MODULAR'
    wave_type_dict = {"sine": 0,
                      "square": 1,
                      "sawtooth": 2}

    filter_type_dict = {"low_pass": 0,
                        "high_pass": 1}

    semitones_max_offset: int = 24
    middle_c_freq: float = 261.6255653005985
    min_amp: float = 0.05
    max_amp: float = 1
    min_mod_index: float = 0.01
    max_mod_index: float = 0.3
    min_lfo_freq: float = 0.5
    max_lfo_freq: float = 20
    min_filter_freq: float = 100
    max_filter_freq: float = 8000
    min_resonance_val: float = 0.01
    max_resonance_val: float = 10
    min_amount_tremolo: float = 0.05
    max_amount_tremolo: float = 1

    fixed_note_off: bool = True
    note_off_time: float = 3.0

    # non-active operation defaults
    non_active_waveform_default = 'sine'
    non_active_freq_default = 0
    non_active_amp_default = 0
    non_active_mod_index_default = 0
    non_active_tremolo_amount_default = 0

    # When predicting oscillator frequency by regression, the defines are used to normalize the output from the model
    margin: float = 200
    # --------------------------------------
    # -----------Modular Synth--------------
    # --------------------------------------
    # Modular Synth attributes:
    num_channels: int = 3
    num_layers: int = 7

    # Seed for random parameters generator
    seed = 2345124

    # Modular synth possible modules from synth_modules.py
    modular_synth_operations = ['osc', 'fm', 'lfo', 'mix', 'filter', 'env_adsr', 'fm_lfo', 'lfo_sine', 'lfo_non_sine',
                                'fm_sine', 'fm_square', 'fm_saw', 'lowpass_filter']

    modular_synth_params = {'osc': ['amp', 'freq', 'waveform'],
                            'lfo_sine': ['active', 'freq'],
                            'lfo_non_sine': ['freq', 'waveform'],
                            'lfo': ['freq', 'waveform'],
                            'fm_lfo': ['active', 'fm_active', 'freq_c', 'waveform', 'mod_index'],
                            'fm': ['freq_c', 'waveform', 'mod_index'],
                            'fm_sine': ['active', 'fm_active', 'amp_c', 'freq_c', 'mod_index'],
                            'fm_square': ['active', 'fm_active', 'amp_c', 'freq_c', 'mod_index'],
                            'fm_saw': ['active', 'fm_active', 'amp_c', 'freq_c', 'mod_index'],
                            'mix': ['active'],
                            'filter': ['filter_freq', 'filter_type'],
                            'lowpass_filter': ['filter_freq', 'resonance'],
                            'env_adsr': ['attack_t', 'decay_t', 'sustain_t', 'sustain_level', 'release_t'],
                            'amplitude_shape': ['envelope', 'attack_t', 'decay_t', 'sustain_t', 'sustain_level',
                                                'release_t'],
                            'tremolo': ['amount', 'active']}

    def __post_init__(self):
        self.wave_type_dic_inv = {v: k for k, v in self.wave_type_dict.items()}
        self.filter_type_dic_inv = {v: k for k, v in self.filter_type_dict.items()}

        # build a list of possible frequencies
        self.semitones_list = [*range(-self.semitones_max_offset, self.semitones_max_offset + 1)]
        self.osc_freq_list = [self.middle_c_freq * (2 ** (1 / 12)) ** x for x in self.semitones_list]
        self.osc_freq_dic = {round(key, 4): value for value, key in enumerate(self.osc_freq_list)}
        self.osc_freq_dic_inv = {v: k for k, v in self.osc_freq_dic.items()}
        self.oscillator_freq = self.osc_freq_list[-1] + self.margin

        self.all_params_presets = {
            'lfo': {'freq': np.asarray([0.5] + [k+1 for k in range(self.max_lfo_freq)])},
            'fm': {'freq_c': np.asarray(self.osc_freq_list),
                   'mod_index': np.linspace(0, self.max_mod_index, 16)},
            'filter': {'filter_freq': np.asarray([100*1.4**k for k in range(14)])}
        }


def configure_experiment(exp_name: str, dataset_name: str):

    project_root = os.path.join(EXP_ROOT, 'current', exp_name, '')

    if os.path.isdir(project_root):
        # overwrite = input(colored(f"Folder {project_root} already exists. Overwrite previous experiment (Y/N)?"
        #                           f"\n\tThis will delete all files related to the previous run!",
        #                           'yellow'))
        # if overwrite.lower() != 'y':
        #     print('Exiting...')
        #     exit()
        # else:
            print("Deleting previous experiment...")
            rmtree(project_root)

    cfg = Config(project_root)
    synth_cfg = SynthConfig()
    dataset_cfg = DatasetConfig(dataset_name)
    model_cfg = ModelConfig()

    config_dump_path = os.path.join(cfg.project_root, 'config_dump', '')
    os.makedirs(config_dump_path, exist_ok=True)

    for fname, t_cfg in zip(['general', 'model', 'synth', 'dataset'], [cfg, model_cfg, synth_cfg, dataset_cfg]):

        config_output_path = os.path.join(config_dump_path, fname + '.json')
        cfg_dict = asdict(t_cfg)

        with open(config_output_path, 'w') as f:
            json.dump(cfg_dict, f)

    return cfg, model_cfg, synth_cfg, dataset_cfg
