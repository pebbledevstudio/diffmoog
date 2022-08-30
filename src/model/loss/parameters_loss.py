import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from synth.synth_constants import SynthConstants


class ParametersLoss:
    """This loss compares target and predicted parameters of the modular synthesizer"""

    def __init__(self, loss_type: str, synth_structure: SynthConstants, device='cuda:0'):
        self.synth_structure = synth_structure
        self.device = device
        if loss_type == 'L1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'L2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("unknown loss type")

        self.cls_loss = nn.CrossEntropyLoss()

    def call(self, predicted_parameters_dict, target_parameters_dict):
        """ execute parameters loss computation between two parameter sets

                Args:
                  :param predicted_parameters_dict: predicted audio parameters
                  :param target_parameters_dict: target audio parameters
                """
        total_loss = 0.0
        loss_dict = {}
        for key in predicted_parameters_dict.keys():
            operation = predicted_parameters_dict[key]['operation']
            predicted_parameters = predicted_parameters_dict[key]['parameters']
            target_parameters = target_parameters_dict[key]['parameters'].copy()

            for param in predicted_parameters.keys():

                if param == 'waveform':
                    waveform_list = [self.synth_structure.wave_type_dict[waveform] for waveform
                                     in target_parameters[param]]
                    target_parameters[param] = torch.tensor(waveform_list)

                elif param == 'filter_type':
                    filter_type_list = [self.synth_structure.filter_type_dict[filter_type] for
                                        filter_type in target_parameters[param]]
                    target_parameters[param] = torch.tensor(filter_type_list)

                elif param in ['active', 'fm_active']:
                    active_list = [0 if is_active else 1 for is_active in target_parameters[param]]
                    target_parameters[param] = torch.tensor(active_list)

                target_parameters[param] = target_parameters[param].type(torch.FloatTensor).to(self.device)
                if predicted_parameters[param].dim() > 1 and predicted_parameters[param].shape[0] == 1:
                    predicted_parameters[param] = predicted_parameters[param].squeeze(dim=0)
                else:
                    predicted_parameters[param] = predicted_parameters[param].squeeze()

                if target_parameters[param].dim() > 1:
                    target_parameters[param] = target_parameters[param].squeeze(dim=0)

                if param in ['waveform', 'filter_type', 'active', 'fm_active']:
                    target_parameters[param] = target_parameters[param].type(torch.LongTensor).to(self.device)
                    loss = self.cls_loss(predicted_parameters[param], target_parameters[param])
                else:
                    loss = self.criterion(predicted_parameters[param], target_parameters[param])

                total_loss += loss

                loss_dict[f"{key}_{operation}_{param}"] = loss

        return total_loss, loss_dict
