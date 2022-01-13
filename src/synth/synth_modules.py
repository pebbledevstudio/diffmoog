#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:41:38 2021

@author: Moshe Laufer, Noy Uzrad
"""
import torch
from config import PI, TWO_PI, DEBUG_MODE
import matplotlib.pyplot as plt
# import simpleaudio as sa
import helper
import julius
from synth.synth_config import *


class SynthModules:
    def __init__(self, num_sounds=1):
        self.sample_rate = SAMPLE_RATE
        self.sig_duration = SIGNAL_DURATION_SEC
        self.time_samples = torch.linspace(0, self.sig_duration, steps=int(self.sample_rate * self.sig_duration),
                                           requires_grad=True)
        self.modulation_time = torch.linspace(0, self.sig_duration, steps=self.sample_rate)
        self.modulation = 0
        self.signal = torch.zeros(size=(num_sounds, self.time_samples.shape[0]), dtype=torch.float32,
                                  requires_grad=True)

        self.device = helper.get_device()
        self.time_samples = helper.move_to(self.time_samples, self.device)
        self.modulation_time = helper.move_to(self.modulation_time, self.device)
        self.signal = helper.move_to(self.signal, self.device)
        # self.room_impulse_responses = torch.load('rir_for_reverb_no_amp')

    def oscillator(self, amp, freq, waveform, phase=0, num_sounds=1):
        """Creates a basic oscillator.

            Retrieves a waveform shape and attributes, and construct the respected signal

            Args:
                self: Self object
                amp: Amplitude in range [0, 1]
                freq: Frequency in range [0, 22000]
                phase: Phase in range [0, 2pi], default is 0
                waveform: a string, one of ['sine', 'square', 'triangle', 'sawtooth'] or a probability vector
                num_sounds: number of sounds to process

            Returns:
                A torch with the constructed signal

            Raises:
                ValueError: Provided variables are out of range
                :rtype: object
            """

        self.signal_values_sanity_check(amp, freq, waveform)
        t = self.time_samples
        # oscillator = torch.zeros_like(t, requires_grad=True)

        oscillator_tensor = torch.tensor((), requires_grad=True).to(helper.get_device())
        first_time = True
        for i in range(num_sounds):

            if num_sounds == 1:
                freq_float = freq
            else:
                freq_float = freq[i]

            sine_wave = amp * torch.sin(TWO_PI * freq_float * t + phase)
            square_wave = amp * torch.sign(torch.sin(TWO_PI * freq_float * t + phase))
            triangle_wave = (2 * amp / PI) * torch.arcsin(torch.sin((TWO_PI * freq_float * t + phase)))
            # triangle_wave = amp * torch.sin(TWO_PI * freq_float * t + phase)
            sawtooth_wave = 2 * (t * freq_float - torch.floor(0.5 + t * freq_float))  # Sawtooth closed form
            # Phase shift (by normalization to range [0,1] and modulo operation)
            sawtooth_wave = (((sawtooth_wave + 1) / 2) + phase / TWO_PI) % 1
            sawtooth_wave = amp * (sawtooth_wave * 2 - 1)  # re-normalization to range [-amp, amp]

            if isinstance(waveform, str):
                if waveform == 'sine':
                    oscillator = sine_wave
                elif waveform == 'square':
                    oscillator = square_wave
                elif waveform == 'triangle':
                    oscillator = triangle_wave
                elif waveform == 'sawtooth':
                    oscillator = sawtooth_wave

            else:
                waveform_probabilities = waveform[i]
                oscillator = waveform_probabilities[0] * sine_wave \
                             + waveform_probabilities[1] * square_wave \
                             + waveform_probabilities[2] * triangle_wave \
                             + waveform_probabilities[3] * sawtooth_wave

            if first_time:
                oscillator_tensor = torch.cat((oscillator_tensor, oscillator), dim=0).unsqueeze(dim=0)
                first_time = False
            else:
                oscillator = oscillator.unsqueeze(dim=0)
                oscillator_tensor = torch.cat((oscillator_tensor, oscillator), dim=0)

        return oscillator_tensor

    def mix_signal(self, new_signal, factor):
        """Signal superposition. factor balances the mix
        1 - original signal only, 0 - new signal only, 0.5 evenly balanced. """
        if factor < 0 or factor > 1:
            raise ValueError("Provided factor value is out of range [0, 1]")
        self.signal = factor * self.signal + (1 - factor) * new_signal

    def oscillator_fm(self, amp_c, freq_c, waveform, mod_index, modulator, num_sounds=1):
        """Basic oscillator with FM modulation

            Creates an oscillator and modulates its frequency by a given modulator

            Args:
                self: Self object
                amp_c: Amplitude in range [0, 1]
                freq_c: Frequency in range [0, 22000]
                waveform: One of [sine, square, triangle, sawtooth]
                mod_index: Modulation index, which affects the amount of modulation
                modulator: Modulator signal, to affect carrier frequency
                num_sounds: number of sounds to process

            Returns:
                A torch with the constructed FM signal

            Raises:
                ValueError: Provided variables are out of range
            """

        self.signal_values_sanity_check(amp_c, freq_c, waveform)
        t = self.time_samples
        oscillator_tensor = torch.tensor((), requires_grad=True).to(helper.get_device())
        first_time = True
        for i in range(num_sounds):
            if num_sounds == 1:
                amp_float = amp_c
                mod_index_float = mod_index
                freq_float = freq_c
                input_signal_cur = modulator
            else:
                amp_float = amp_c[i]
                mod_index_float = mod_index[i]
                freq_float = freq_c[i]
                input_signal_cur = modulator[i]

            fm_sine_wave = amp_float * torch.sin(TWO_PI * freq_float * t + mod_index_float * input_signal_cur)
            fm_square_wave = amp_float * torch.sign(torch.sin(TWO_PI * freq_float * t + mod_index_float * input_signal_cur))
            # fm_triangle_wave = (2 * amp_float / PI) * torch.arcsin(torch.sin((TWO_PI * freq_float * t + mod_index_float * input_signal_cur)))
            fm_triangle_wave = amp_float * torch.sin(TWO_PI * freq_float * t + mod_index_float * input_signal_cur)

            fm_sawtooth_wave = 2 * (t * freq_float - torch.floor(0.5 + t * freq_float))
            fm_sawtooth_wave = (((fm_sawtooth_wave + 1) / 2) + mod_index_float * input_signal_cur / TWO_PI) % 1
            fm_sawtooth_wave = amp_float * (fm_sawtooth_wave * 2 - 1)

            if isinstance(waveform, list):
                waveform = waveform[i]

            if isinstance(waveform, str):
                if waveform == 'sine':
                    oscillator = fm_sine_wave
                elif waveform == 'square':
                    oscillator = fm_square_wave
                elif waveform == 'triangle':
                    oscillator = fm_triangle_wave
                elif waveform == 'sawtooth':
                    oscillator = fm_sawtooth_wave

            else:
                if num_sounds == 1:
                    waveform_probabilities = waveform

                elif torch.is_tensor(waveform):
                    waveform_probabilities = waveform[i]

                oscillator = waveform_probabilities[0] * fm_sine_wave \
                             + waveform_probabilities[1] * fm_square_wave \
                             + waveform_probabilities[2] * fm_sawtooth_wave

            if first_time:
                if num_sounds == 1:
                    oscillator_tensor = oscillator
                else:
                    oscillator_tensor = torch.cat((oscillator_tensor, oscillator), dim=0).unsqueeze(dim=0)
                    first_time = False
            else:
                oscillator = oscillator.unsqueeze(dim=0)
                oscillator_tensor = torch.cat((oscillator_tensor, oscillator), dim=0)

        return oscillator_tensor

    def am_modulation(self, amp_c, freq_c, amp_m, freq_m, final_max_amp, waveform):
        """AM modulation

            Modulates the amplitude of a carrier signal with a sine modulator
            see https://en.wikipedia.org/wiki/Amplitude_modulation

            Args:
                self: Self object
                amp_c: Amplitude of carrier in range [0, 1]
                freq_c: Frequency of carrier in range [0, 22000]
                amp_m: Amplitude of modulator in range [0, 1]
                freq_m: Frequency of modulator in range [0, 22000]
                final_max_amp: The final maximum amplitude of the modulated signal
                waveform: One of [sine, square, triangle, sawtooth]

            Returns:
                A torch with the constructed AM signal

            Raises:
                ValueError: Provided variables are out of range
                ValueError: modulation index > 1. Amplitude values must obey amp_m < amp_c
                # todo add documentation for sensible frequency values
            """
        self.signal_values_sanity_check(amp_m, freq_m, waveform)
        self.signal_values_sanity_check(amp_c, freq_c, waveform)
        modulation_index = amp_m / amp_c
        if modulation_index > 1:
            raise ValueError("Provided amplitudes results modulation index > 1, and yields over-modulation ")
        if final_max_amp < 0 or final_max_amp > 1:
            raise ValueError("Provided final max amplitude is not in range [0, 1]")
        # todo: add restriction freq_c >> freq_m

        t = self.time_samples
        dc = 1
        carrier = SynthModules()
        carrier.oscillator(amp=amp_c, freq=freq_c, phase=0, waveform=waveform)
        modulator = amp_m * torch.sin(TWO_PI * freq_m * t)
        am_signal = (dc + modulator / amp_c) * carrier.signal
        normalized_am_signal = (final_max_amp / (amp_c + amp_m)) * am_signal
        self.signal = normalized_am_signal

    def am_modulation_by_input_signal(self, input_signal, modulation_factor, amp_c, freq_c, waveform):
        """AM modulation by an input signal

            Modulates the amplitude of a carrier signal with a provided input signal
            see https://en.wikipedia.org/wiki/Amplitude_modulation

            Args:
                self: Self object
                input_signal: Input signal to be used as modulator
                modulation_factor: factor to be multiplied by modulator, in range [0, 1]
                amp_c: Amplitude of carrier in range [0, 1]
                freq_c: Frequency of carrier in range [0, 22000]
                waveform: Waveform of carrier. One of [sine, square, triangle, sawtooth]

            Returns:
                A torch with the constructed AM signal

            Raises:
                ValueError: Provided variables are inappropriate
                ValueError: Resulted Amplitude is out of range [-1, 1]
            """
        self.signal_values_sanity_check(amp_c, freq_c, waveform)
        carrier = SynthModules()
        carrier.oscillator(amp=1, freq=freq_c, phase=0, waveform=waveform)
        modulated_amplitude = (amp_c + modulation_factor * input_signal)
        if torch.max(modulated_amplitude).item() > 1 or torch.min(modulated_amplitude).item() < -1:
            raise ValueError("AM modulation resulted amplitude out of range [-1, 1].")
        self.signal = modulated_amplitude * carrier.signal

    def adsr_envelope(self, input_signal, attack_t, decay_t, sustain_t, sustain_level, release_t, num_sounds=1):
        """Apply an ADSR envelope to the signal

            builds the ADSR shape and multiply by the signal

            Args:
                self: Self object
                input_signal: target signal to apply adsr
                attack_t: Length of attack in seconds. Time to go from 0 to 1 amplitude.
                decay_t: Length of decay in seconds. Time to go from 1 amplitude to sustain level.
                sustain_t: Length of sustain in seconds, with sustain level amplitude
                sustain_level: Sustain volume level
                release_t: Length of release in seconds. Time to go ftom sustain level to 0 amplitude
                num_sounds: number of sounds to process

            Raises:
                ValueError: Provided ADSR timings are not the same as the signal length
            """
        if num_sounds == 1:
            if attack_t + decay_t + sustain_t + release_t > self.sig_duration:
                raise ValueError("Provided ADSR durations exceeds signal duration")

        else:
            for i in range(num_sounds):
                if attack_t[i] + decay_t[i] + sustain_t[i] + release_t[i] > self.sig_duration:
                    raise ValueError("Provided ADSR durations exceeds signal duration")

        if num_sounds == 1:
            attack_num_samples = int(self.sample_rate * attack_t)
            decay_num_samples = int(self.sample_rate * decay_t)
            sustain_num_samples = int(self.sample_rate * sustain_t)
            release_num_samples = int(self.sample_rate * release_t)
        else:
            attack_num_samples = [torch.floor(torch.tensor(self.sample_rate * attack_t[k])) for k in range(num_sounds)]
            decay_num_samples = [torch.floor(torch.tensor(self.sample_rate * decay_t[k])) for k in range(num_sounds)]
            sustain_num_samples = [torch.floor(torch.tensor(self.sample_rate * sustain_t[k])) for k in range(num_sounds)]
            release_num_samples = [torch.floor(torch.tensor(self.sample_rate * release_t[k])) for k in range(num_sounds)]
            attack_num_samples = torch.stack(attack_num_samples)
            decay_num_samples = torch.stack(decay_num_samples)
            sustain_num_samples = torch.stack(sustain_num_samples)
            release_num_samples = torch.stack(release_num_samples)

        if num_sounds > 1:
            # todo: change the check with sustain_level[0]
            if torch.is_tensor(sustain_level[0]):
                sustain_level = [sustain_level[i] for i in range(num_sounds)]
                sustain_level = torch.stack(sustain_level)
            else:
                sustain_level = [sustain_level[i] for i in range(num_sounds)]

        enveloped_signal_tensor = torch.tensor((), requires_grad=True).to(helper.get_device())
        first_time = True
        for i in range(num_sounds):

            if num_sounds == 1:
                attack = torch.linspace(0, 1, attack_num_samples)
                decay = torch.linspace(1, sustain_level, decay_num_samples)
                sustain = torch.full((sustain_num_samples,), sustain_level)
                release = torch.linspace(sustain_level, 0, release_num_samples)
            else:
                attack = torch.linspace(0, 1, int(attack_num_samples[i].item()), device=helper.get_device())
                decay = torch.linspace(1, sustain_level[i], int(decay_num_samples[i]), device=helper.get_device())
                sustain = torch.full((int(sustain_num_samples[i].item()),), sustain_level[i], device=helper.get_device())
                release = torch.linspace(sustain_level[i], 0, int(release_num_samples[i].item()), device=helper.get_device())

                # todo: make sure ADSR behavior is differentiable. linspace has to know to get tensors
                # attack_mod = helper.linspace(torch.tensor(0), torch.tensor(1), attack_num_samples[i])
                # decay_mod = helper.linspace(torch.tensor(1), sustain_level[i], decay_num_samples[i])
                # sustain_mod = torch.full((sustain_num_samples[i],), sustain_level[i])
                # release_mod = helper.linspace(sustain_num_samples[i], torch.tensor(0), release_num_samples[i])

            envelope = torch.cat((attack, decay, sustain, release))
            envelope = helper.move_to(envelope, self.device)

            # envelope = torch.cat((attack_mod, decay_mod, sustain_mod, release_mod))

            envelope_len = envelope.shape[0]
            signal_len = self.time_samples.shape[0]
            if envelope_len <= signal_len:
                padding = torch.zeros((signal_len - envelope_len), device=helper.get_device())
                envelope = torch.cat((envelope, padding))
            else:
                raise ValueError("Envelope length exceeds signal duration")

            if torch.is_tensor(input_signal) and num_sounds > 1:
                signal_to_shape = input_signal[i]
            else:
                signal_to_shape = input_signal

            enveloped_signal = signal_to_shape * envelope

            if first_time:
                if num_sounds == 1:
                    enveloped_signal_tensor = enveloped_signal
                else:
                    enveloped_signal_tensor = torch.cat((enveloped_signal_tensor, enveloped_signal), dim=0).unsqueeze(dim=0)
                    first_time = False
            else:
                enveloped = enveloped_signal.unsqueeze(dim=0)
                enveloped_signal_tensor = torch.cat((enveloped_signal_tensor, enveloped), dim=0)

            if DEBUG_MODE:
                plt.plot(envelope.cpu())
                plt.plot(self.signal.cpu())
                plt.show()

        return enveloped_signal_tensor

    def filter(self, input_signal, filter_freq, filter_type, num_sounds=1):
        """Apply an ADSR envelope to the signal

            builds the ADSR shape and multiply by the signal

            Args:
                self: Self object
                :param input_signal: 1D or 2D array or tensor to apply filter along rows
                :param filter_type: one of ['low_pass', 'high_pass', 'band_pass']
                :param filter_freq: corner or central frequency
                :param num_sounds: number of sounds in the input

            Raises:
                none

            """
        filtered_signal_tensor = torch.tensor((), requires_grad=True).to(helper.get_device())
        first_time = True
        for i in range(num_sounds):
            if num_sounds == 1:
                filter_frequency = filter_freq
            elif num_sounds > 1:
                filter_frequency = filter_freq[i]

            if torch.is_tensor(filter_frequency):
                filter_frequency = helper.move_to(filter_frequency, "cpu")
            high_pass_signal = self.high_pass(input_signal[i], cutoff_freq=filter_frequency, index=i)
            low_pass_signal = self.low_pass(input_signal[i], cutoff_freq=filter_frequency, index=i)

            if isinstance(filter_type, list):
                filter_type = filter_type[i]

            if isinstance(filter_type, str):
                if filter_type == 'high_pass':
                    filtered_signal = high_pass_signal
                elif filter_type == 'low_pass':
                    filtered_signal = low_pass_signal

            else:
                if num_sounds == 1:
                    filter_type_probabilities = filter_type
                else:
                    filter_type_probabilities = filter_type[i]

                filter_type_probabilities = filter_type_probabilities.cpu()
                filtered_signal = filter_type_probabilities[0] * high_pass_signal \
                                  + filter_type_probabilities[1] * low_pass_signal
                filtered_signal = filtered_signal.to(helper.get_device())

            if first_time:
                if num_sounds == 1:
                    filtered_signal_tensor = filtered_signal
                else:
                    filtered_signal_tensor = torch.cat((filtered_signal_tensor, filtered_signal), dim=0).unsqueeze(dim=0)
                    first_time = False
            else:
                filtered_signal = filtered_signal.unsqueeze(dim=0)
                filtered_signal_tensor = torch.cat((filtered_signal_tensor, filtered_signal), dim=0)

        return filtered_signal_tensor

    def low_pass(self, input_signal, cutoff_freq, q=0.707, index=0):
        # filtered_waveform = taF.lowpass_biquad(input_signal, self.sample_rate, cutoff_freq, q)
        # filtered_waveform = julius.lowpass_filter(input_signal, cutoff_freq.item()/44100)
        if cutoff_freq == 0:
            return input_signal
        else:
            filtered_waveform_new = julius.lowpass_filter_new(input_signal, cutoff_freq / 44100)
            return filtered_waveform_new

    def high_pass(self, input_signal, cutoff_freq, q=0.707, index=0):
        # filtered_waveform = julius.lowpass_filter(input_signal, cutoff_freq.item()/44100)
        # filtered_waveform_new = julius.highpass_filter(input_signal, cutoff_freq/44100)
        if cutoff_freq == 0:
            return input_signal
        else:
            filtered_waveform_new = julius.highpass_filter_new(input_signal, cutoff_freq / 44100)
            return filtered_waveform_new

    @staticmethod
    # todo: remove all except list instances
    def signal_values_sanity_check(amp, freq, waveform):
        """Check signal properties are reasonable."""
        if isinstance(freq, float):
            if freq < 0 or freq > 20000:
                raise ValueError("Provided frequency is not in range [0, 20000]")
        elif isinstance(freq, list):
            if any(element < 0 or element > 2000 for element in freq):
                raise ValueError("Provided frequency is not in range [0, 20000]")
        if isinstance(amp, int):
            if amp < 0 or amp > 1:
                raise ValueError("Provided amplitude is not in range [0, 1]")
        elif isinstance(amp, list):
            if any(element < 0 or element > 1 for element in amp):
                raise ValueError("Provided amplitude is not in range [0, 1]")
        if isinstance(waveform, str):
            if not any(x in waveform for x in ['sine', 'square', 'triangle', 'sawtooth']):
                raise ValueError("Unknown waveform provided")


"""
Reverb implementation - Currently not working as expected
So it is not used

    def reverb(self, size, dry_wet):
        if size not in [1, 2, 3, 4, 5, 6]:
            raise ValueError("reverb size must be an int in range [1, 6]")
        if dry_wet < 0 or dry_wet > 1:
            raise ValueError("Provided dry/wet value is out of range [0, 1]")

        if size == 1:
            response_name = 'rir0'
        elif size == 2:
            response_name = 'rir20'
        elif size == 3:
            response_name = 'rir40'
        elif size == 4:
            response_name = 'rir60'
        elif size == 5:
            response_name = 'rir80'
        elif size == 6:
            response_name = 'rir100'
        else:
            response_name = 'rir0'

        signal_clean = self.signal
        room_impulse_response = self.room_impulse_responses[response_name]

        # room_response_start_index = 0
        # room_response_end_index = room_impulse_response.size()[0] - 1
        # signal_start_index = 0
        # signal_end_index = signal_clean.size()[0] - 1
        # while room_impulse_response[room_response_start_index] == 0:
        #     room_response_start_index = room_response_start_index+1
        # while room_impulse_response[room_response_end_index] == 0:
        #     room_response_end_index = room_response_end_index-1
        # while signal_clean[signal_start_index] == 0:
        #     signal_start_index = signal_start_index+1
        # while signal_clean[signal_end_index] == 0:
        #     signal_end_index = signal_end_index-1
        # print(room_impulse_response.shape)
        # print(signal_clean.shape)
        # signal_clean = signal_clean[signal_start_index:signal_end_index]
        # room_impulse_response = room_impulse_response[room_response_start_index:room_response_end_index]

        kernel_size = room_impulse_response.shape[0]
        signal_size = signal_clean.shape[0]

        print("kersize", kernel_size)
        print("sigsize", signal_size)
        print(room_impulse_response.shape)
        print(signal_clean.shape)

        room_impulse_response = room_impulse_response.unsqueeze(0).unsqueeze(0)
        signal_clean = signal_clean.unsqueeze(0).unsqueeze(0)

        signal_reverb = F.conv1d(signal_clean, room_impulse_response, bias=None, stride=1)

        signal_reverb = torch.squeeze(signal_reverb)
        signal_clean = torch.squeeze(signal_clean)

        print("sig rev shape", signal_reverb.shape)
        print(signal_reverb.shape[0])

        if signal_reverb.shape[0] > signal_clean.shape[0]:
            padding = torch.zeros(signal_reverb.shape[0] - signal_clean.shape[0])
            signal_clean = torch.cat((signal_clean, padding))
        else:
            padding = torch.zeros(signal_clean.shape[0] - signal_reverb.shape[0])
            signal_reverb = torch.cat((signal_reverb, padding))

        self.signal = dry_wet * signal_reverb + (1 - dry_wet) * signal_clean
        self.signal = torch.squeeze(self.signal)
"""

if __name__ == "__main__":
    print(len(OSC_FREQ_LIST))
    a = SynthModules()
    b = SynthModules()
    b.oscillator(1, 5, 0, 'sine')
    a.oscillator_fm(amp_c=1, freq_c=440, waveform='sine', mod_index=10, modulator=b.signal)
    # a.oscillator(amp=1, freq=100, phase=0, waveform='sine')
    # a.adsr_envelope(attack_t=0.5, decay_t=0, sustain_t=0.5, sustain_level=0.5, release_t=0)
    # plt.plot(a.signal)
    # plt.show
    # play_obj = sa.play_buffer(a.signal.numpy(), num_channels=1, bytes_per_sample=4, sample_rate=a.sample_rate)
    # play_obj.wait_done()
    # b = Signal()
    # a.am_modulation(amp_c=1, freq_c=4, amp_m=0.3, freq_m=0, final_max_amp=0.5, waveform='sine')
    # b.am_modulation_by_input_signal(a.data, modulation_factor=1, amp_c=0.5, freq_c=40, waveform='triangle')
    plt.plot(a.signal)
    # plt.plot(b.data)
    # torch.tensor(0)
    # print(torch.sign(torch.tensor(0)))
    # # b.fm_modulation_by_input_signal(a.data, 440, 1, 10, 'sawtooth')
    # # plt.plot(b.data)
    # plt.show()
    #
    # # plt.plot(a.data)
    # a.low_pass(1000)
    # play_obj = sa.play_buffer(b.data.numpy(), 1, 4, b.sample_rate)
    # play_obj.wait_done()
    # plt.plot(a.data)
    #
    # def fm_modulation(self, amp_mod, fm, fc, Ac, waveform):
    # a.fm_modulation(1, 3, 5, 1, 'tri')
    # print(a.data)
    # plt.plot(a.data)
    #
    # plt.show()