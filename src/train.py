import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from src.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DEBUG_MODE
from src.ai_synth_dataset import AiSynthDataset
from src.config import PARAMETERS_FILE, AUDIO_DIR
from synth_model import SynthNetwork
from sound_generator import SynthBasicFlow
import synth
import helper


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, optimizer_arg, device_arg):
    for signal_mel_spectrogram, target_params_dic in data_loader:

        batch_size = signal_mel_spectrogram.shape[0]
        classification_target_params = target_params_dic['classification_params']
        regression_target_parameters = target_params_dic['regression_params']
        helper.map_classification_params_to_ints(classification_target_params)

        classification_target_params = helper.move_to(classification_target_params, device_arg)
        regression_target_parameters = helper.move_to(regression_target_parameters, device_arg)
        signal_mel_spectrogram = helper.move_to(signal_mel_spectrogram, device_arg)

        # todo: normalize/standardize/rescale target parameters from 0 to 1. use log scale for frequencies
        if DEBUG_MODE:
            helper.plot_spectrogram(signal_mel_spectrogram[0][0].cpu(), title="MelSpectrogram - torchaudio", ylabel='mel freq')
        output_dic = model(signal_mel_spectrogram)

        # Infer predictions
        predicted_dic = {}
        for param in synth.CLASSIFICATION_PARAM_LIST:
            predicted_dic[param] = torch.argmax(output_dic[param], dim=1)
        for index, param in enumerate(synth.REGRESSION_PARAM_LIST):
            predicted_dic[param] = output_dic['regression_params'][:, index]

        helper.map_classification_params_from_ints(predicted_dic)
        # todo: force the model to predict values from defined ranges - check below line if correct
        helper.clamp_regression_params(predicted_dic)

        # Init criteria
        criterion_osc1_freq = criterion_osc1_wave = criterion_lfo1_wave \
            = criterion_osc2_freq = criterion_osc2_wave = criterion_lfo2_wave \
            = criterion_filter_type \
            = nn.CrossEntropyLoss()
        criterion_regression_params = criterion_spectrogram = nn.MSELoss()

        loss_spectrogram_total = 0
        current_predicted_dic = {}
        # todo: refactor code. try to implement SynthBasicFlow in matrix, to prevent for loop.
        #  compute all spectrogram loss all at once using mean function
        for i in range(batch_size):
            for key, value in predicted_dic.items():
                if torch.is_tensor(predicted_dic[key][i]):
                    current_predicted_dic[key] = predicted_dic[key][i].item()
                else:
                    current_predicted_dic[key] = predicted_dic[key][i]

            synth_obj = SynthBasicFlow(parameters_dict=current_predicted_dic)

            predicted_mel_spec_sound_signal = helper.mel_spectrogram_transform(synth_obj.signal)
            predicted_mel_spec_sound_signal = helper.move_to(predicted_mel_spec_sound_signal, device_arg)
            # todo: refactor code. use unsqueeze instead of '0'.
            current_loss_spectrogram = criterion_spectrogram(predicted_mel_spec_sound_signal,
                                                             signal_mel_spectrogram[i][0])
            loss_spectrogram_total = loss_spectrogram_total + current_loss_spectrogram

        loss_spectrogram_total = loss_spectrogram_total / batch_size

        loss_osc1_freq = criterion_osc1_freq(output_dic['osc1_freq'], classification_target_params['osc1_freq'])
        loss_osc1_wave = criterion_osc1_wave(output_dic['osc1_wave'], classification_target_params['osc1_wave'])
        loss_lfo1_wave = criterion_lfo1_wave(output_dic['lfo1_wave'], classification_target_params['lfo1_wave'])
        loss_osc2_freq = criterion_osc2_freq(output_dic['osc2_freq'], classification_target_params['osc2_freq'])
        loss_osc2_wave = criterion_osc2_wave(output_dic['osc2_wave'], classification_target_params['osc2_wave'])
        loss_lfo2_wave = criterion_lfo2_wave(output_dic['lfo2_wave'], classification_target_params['lfo2_wave'])
        loss_filter_type = \
            criterion_filter_type(output_dic['filter_type'], classification_target_params['filter_type'])

        # todo: refactor code. the code gets dictionary of tensors (regression_target_parameters) and return 2d tensor
        regression_target_parameters_tensor = torch.empty((len(regression_target_parameters['osc1_amp']), 1))
        regression_target_parameters_tensor = helper.move_to(regression_target_parameters_tensor, device_arg)
        for key, value in regression_target_parameters.items():
            regression_target_parameters_tensor = \
                torch.cat([regression_target_parameters_tensor, regression_target_parameters[key].unsqueeze(dim=1)],
                          dim=1)
        regression_target_parameters_tensor = regression_target_parameters_tensor[:, 1:]
        regression_target_parameters_tensor = regression_target_parameters_tensor.float()

        loss_classification_params = \
            loss_osc1_freq + loss_osc1_wave + loss_lfo1_wave + \
            loss_osc2_freq + loss_osc2_wave + loss_lfo2_wave + \
            loss_filter_type

        loss_regression_params = \
            criterion_regression_params(output_dic['regression_params'], regression_target_parameters_tensor)

        # todo: balance losses so they are in the same range
        loss = loss_classification_params + loss_regression_params + loss_spectrogram_total

        # backpropogate error and update wights
        optimizer_arg.zero_grad()
        loss.backward()
        optimizer_arg.step()

    print(f"loss: {loss.item()}")
    if DEBUG_MODE:
        print("osc1_freq",
              torch.argmax(output_dic['osc1_freq'], dim=1), classification_target_params['osc1_freq'])
        print("osc1_wave",
              torch.argmax(output_dic['osc1_wave'], dim=1), classification_target_params['osc1_wave'])
        print("lfo1_wave",
              torch.argmax(output_dic['lfo1_wave'], dim=1), classification_target_params['lfo1_wave'])
        print("osc2_freq",
              torch.argmax(output_dic['osc2_freq'], dim=1), classification_target_params['osc2_freq'])
        print("osc2_wave",
              torch.argmax(output_dic['osc2_wave'], dim=1), classification_target_params['osc2_wave'])
        print("lfo2_wave",
              torch.argmax(output_dic['lfo2_wave'], dim=1), classification_target_params['lfo2_wave'])
        print("filter_type",
              torch.argmax(output_dic['filter_type'], dim=1), classification_target_params['filter_type'])
        print("regression_params",
              output_dic['regression_params'], regression_target_parameters_tensor)


def train(model, data_loader, optimiser_arg, device_arg, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, optimiser_arg, device_arg)
        print("--------------------------------------")
    print("Finished training")


if __name__ == "__main__":
    device = helper.get_device()

    ai_synth_dataset = AiSynthDataset(PARAMETERS_FILE,
                                      AUDIO_DIR,
                                      helper.mel_spectrogram_transform,
                                      synth.SAMPLE_RATE,
                                      device)
    train_dataloader = create_data_loader(ai_synth_dataset, BATCH_SIZE)

    # construct model and assign it to device
    synth_net = SynthNetwork().to(device)

    # initialize optimizer
    optimiser = torch.optim.Adam(synth_net.parameters(), lr=LEARNING_RATE)

    # train model
    train(synth_net, train_dataloader, optimiser, device, EPOCHS)

    # save model
    torch.save(synth_net.state_dict(), "../trained_models/synth_net.pth")
    print("Trained synth net saved at synth_net.pth")