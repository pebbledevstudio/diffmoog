import os
import torchaudio
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset.ai_synth_dataset import NSynthDataset
from model.lit_module import LitModularSynth
from utils.train_utils import get_project_root, to_numpy_recursive
from IPython.utils.io import capture_output
import torch
from pathlib import Path

root = get_project_root()

### set these variables before evaluation
experiment_name = ''
experiment_version = 0
ckpt_name = ''
ckpt_path = os.path.join(root, 'my_checkpoints', f'exp_{experiment_name}_version_{experiment_version}', ckpt_name)
cfg_path = os.path.join(root, 'configs', '')
dataset_path = os.path.join(root, 'data', '')
using_nsynth_dataset = False # A different step function is used when evaluating using the nsynth_dataset
output_path = os.path.join(root, 'output')
device = 'cpu'
###

if not os.path.exists(output_path):
    os.makedirs(output_path)

cfg = OmegaConf.load(cfg_path)
synth_module = LitModularSynth.load_from_checkpoint(ckpt_path, device=device, map_location=torch.device('cpu'), train_cfg=cfg).to(device)

val_dataset = NSynthDataset(dataset_path)
dataloader = DataLoader(val_dataset, batch_size=64)

all_losses, all_metrics, all_step_artifacts = [], [], []
for batch in tqdm(dataloader):
    batch[0] = batch[0].to(device)
    with torch.no_grad(), capture_output():
        if using_nsynth_dataset:
            loss, step_losses, step_metrics, step_artifacts = synth_module.out_of_domain_step(batch, return_metrics=True)
        else
            loss, step_losses, step_metrics, step_artifacts = synth_module.in_domain_step(batch, return_metrics=True)
    all_losses.append(step_losses)
    all_metrics.append(step_metrics)
    all_step_artifacts.append(step_artifacts)
    break

n = 0
for i, step_artifacts in enumerate(all_step_artifacts[:1]):
    for j, (target_signal, pred_signal) in enumerate(zip(step_artifacts['target_final_signal'], step_artifacts['pred_final_signal'])):
        torchaudio.save(os.path.join(output_path, f'{Path(val_dataset._get_audio_path(n)).stem}({Path(dataset_path).stem})-{Path(ckpt_path).stem}.wav'), pred_signal.reshape((1, -1)), sample_rate=16000)
        n += 1

all_losses = {k: np.mean([to_numpy_recursive(dic[k]) for dic in all_losses]) for k in all_losses[0]}
all_metrics = {k: np.mean([to_numpy_recursive(dic[k]) for dic in all_metrics]) for k in all_metrics[0]}

for title, metrics_dict in zip(['Losses', 'Metrics'], [all_losses, all_metrics]):
    print(f'\n{title}:')
    for metric_name, metric_val in metrics_dict.items():
        print(f"\t{metric_name}: {metric_val}")
    print("************************************************")
