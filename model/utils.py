import torch
import librosa
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None, to_db=True):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    if to_db:
        specgram = librosa.power_to_db(specgram)
    ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest")


def plot_labels(labels, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.imshow(labels, origin="lower", aspect="auto", interpolation="nearest")


class FindUnusedParametersCallback(Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                print(f"Unused param: {name}")
