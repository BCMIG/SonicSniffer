import edt
import string
import os
import torch
import torchaudio
import json
import functools

import torchaudio.functional as F
import torchaudio.transforms as T

from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt


def get_spectrogram(path, n_mels=128):
    speech, sample_rate = torchaudio.load(path)

    n_fft = 1024
    win_length = None
    hop_length = 256
    # hop_length = 512

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )

    mel_specgram = mel_spectrogram(speech)
    # [channels, n_mels, time]
    log_mel_specgram = torchaudio.transforms.AmplitudeToDB()(mel_specgram)
    # [time]: in seconds
    timestamps = torch.linspace(
        0, speech.shape[-1] / sample_rate, log_mel_specgram.shape[-1]
    )

    return log_mel_specgram, sample_rate, timestamps


KEYS = list(string.ascii_lowercase) + [" ", "backspace", "shift", ","]
KEYS_DICT = {key: i for i, key in enumerate(KEYS)}


@functools.cache
def read_keypresses(path):
    with open(path, "r") as f:
        keypresses = json.load(f)
    assert keypresses == sorted(keypresses, key=lambda x: x["timestamp"])
    keys, timestamps, is_keydown = [], [], []
    for event in keypresses:
        # since shift is recorded manually
        key = event["key"].lower()
        assert key in KEYS, f"Unknown key: {key}"
        keys.append(key)
        # in seconds
        timestamps.append(event["timestamp"] / 1000)
        assert event["type"] in ["keydown", "keyup"]
        is_keydown.append(event["type"] == "keydown")

    return keys, timestamps, is_keydown


def reconstruct_keypresses(keys, is_keydown):
    shift = False
    output = []
    for key, is_keydown in zip(keys, is_keydown):
        if key == "shift":
            shift = is_keydown
        elif key == "backspace":
            if is_keydown:
                output = output[:-1]
        else:
            if is_keydown:
                output.append(key.upper() if shift else key)
    return "".join(output)


def generate_labels(spectrogram_timestamps, keys, keypress_timestamps, is_keydown):
    spectrogram_timestamps = spectrogram_timestamps.clone()
    # Make sure the first timestamp is not 0, so that first keydown is registered
    spectrogram_timestamps[0] += 1e-6
    labels = torch.zeros(len(KEYS), len(spectrogram_timestamps))

    merged_timestamps = torch.cat(
        [spectrogram_timestamps, torch.tensor(keypress_timestamps)]
    )
    # negative if spectrogram, non-negative if keypress
    index = torch.tensor(
        [-1] * len(spectrogram_timestamps) + list(range(len(keypress_timestamps)))
    )

    argsort = torch.argsort(merged_timestamps)
    sorted_index = index[argsort]

    # can be used for word segmentation
    space_keyup_idx = []

    keys_down = torch.zeros(len(KEYS), dtype=torch.bool)
    idx = 0
    for keypress_idx in sorted_index:
        if keypress_idx < 0:
            labels[:, idx] = keys_down
            idx += 1
        else:
            key = keys[keypress_idx]
            key_idx = KEYS_DICT[key]
            keys_down[key_idx] = is_keydown[keypress_idx]
            if key == " " and not is_keydown[keypress_idx]:
                space_keyup_idx.append(idx)
    assert idx == len(spectrogram_timestamps)

    return labels, space_keyup_idx


def get_sdf(labels):
    # NOTE: positive if inside keydown
    # [num_keys, time]
    arrays = [labels[i] for i in range(len(KEYS))]
    arrays = [x.numpy() > 0 for x in arrays]
    sdfs = [edt.sdf(x) for x in arrays]
    sdfs = [torch.tensor(x) for x in sdfs]
    sdf = torch.stack(sdfs, dim=0)
    # normalized later in labels
    # make sure to clamp (in case of all zeros/ones)

    return sdf


@functools.cache
def get_specgram_labels(audio_path, keypress_path, sample_rate, num_channels, n_mels):
    # log_mel_specgram: [channels, n_mels, time]
    log_mel_specgram, sample_rate, specgram_timestamps = get_spectrogram(
        audio_path, n_mels
    )
    assert sample_rate == sample_rate
    assert log_mel_specgram.shape[0] == num_channels
    keys, keypress_timestamps, is_keydown = read_keypresses(keypress_path)
    labels, _ = generate_labels(
        specgram_timestamps, keys, keypress_timestamps, is_keydown
    )
    return log_mel_specgram, labels


class SonicSnifferDataset(Dataset):
    def __init__(self, num_samples, data_path):  # , transform=None):
        self.data_path = data_path
        self.audio_files = sorted(
            [x for x in os.listdir(data_path) if x.endswith(".wav")]
        )
        self.keypress_files = sorted(
            [x for x in os.listdir(data_path) if x.endswith(".json")]
        )
        # self.transform = transform

        self.sample_rate = 48000
        self.num_channels = 2
        self.num_samples = num_samples
        self.n_mels = 128

        timestamps = get_spectrogram(os.path.join(self.data_path, self.audio_files[0]))[
            -1
        ]
        print(
            f"{num_samples} num_samples is {num_samples * (timestamps[-1] / len(timestamps))} seconds"
        )

        assert all(
            x.replace(".wav", "") == y.replace(".json", "")
            for x, y in zip(self.audio_files, self.keypress_files)
        )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = os.path.join(self.data_path, self.audio_files[idx])
        keypress_file = os.path.join(self.data_path, self.keypress_files[idx])
        log_mel_specgram, labels = get_specgram_labels(
            audio_file, keypress_file, self.sample_rate, self.num_channels, self.n_mels
        )

        # random num_samples slice
        start_idx = torch.randint(
            0, log_mel_specgram.shape[-1] - self.num_samples, (1,)
        ).item()
        # [channels, n_mels, num_samples]
        sampled_log_mel_specgram = log_mel_specgram[
            :, :, start_idx : start_idx + self.num_samples
        ]
        # z-score normalization
        sampled_log_mel_specgram = (
            sampled_log_mel_specgram - sampled_log_mel_specgram.mean()
        ) / sampled_log_mel_specgram.std()
        # [num_keys, num_samples]
        sampled_labels = labels[:, start_idx : start_idx + self.num_samples]
        sampled_sdf = get_sdf(sampled_labels)
        # clamp to remove infs
        sampled_sdf = torch.clamp(sampled_sdf, -1, 1)
        # normalize sdf to unit length
        sampled_sdf = sampled_sdf / self.num_samples

        assert sampled_log_mel_specgram.shape[0] == self.num_channels
        assert sampled_log_mel_specgram.shape[1] == self.n_mels
        assert (
            sampled_log_mel_specgram.shape[2]
            == sampled_labels.shape[1]
            == self.num_samples
        )
        assert sampled_sdf.shape == sampled_labels.shape

        return sampled_log_mel_specgram, sampled_labels, sampled_sdf

    def estimate_pos_weight(self):
        num_pos = 0
        num_total = 0
        for i in range(len(self)):
            _, labels, _ = self[i]
            num_pos += labels.sum()
            num_total += labels.numel()
        num_neg = num_total - num_pos
        return num_neg / num_pos


def get_dataloaders(num_samples, batch_size, data_dir, num_workers):
    dataset = SonicSnifferDataset(num_samples, data_dir)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    test_size = (dataset_size - train_size) // 2  # 10% for testing
    val_size = dataset_size - train_size - test_size  # 10% for validation

    train_dataset, test_dataset, val_dataset = random_split(
        dataset, [train_size, test_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, val_loader


if __name__ == "__main__":
    data_path = "/run/host/var/home/jason/mnt/SonicSniffer/server/uploads"
    dataset = SonicSnifferDataset(128, data_path)
    # print(f"Estimated pos_weight: {dataset.estimate_pos_weight()}")

    audio_files = sorted([x for x in os.listdir(data_path) if x.endswith(".wav")])
    keypress_files = sorted([x for x in os.listdir(data_path) if x.endswith(".json")])

    log_mel_specgram, sample_rate, specgram_timestamps = get_spectrogram(
        os.path.join(data_path, audio_files[0])
    )
    keys, keypress_timestamps, is_keydown = read_keypresses(
        os.path.join(data_path, keypress_files[0])
    )
    print(f"Reconstructed keypresses: {reconstruct_keypresses(keys, is_keydown)}")
    labels, space_keyup_idx = generate_labels(
        specgram_timestamps, keys, keypress_timestamps, is_keydown
    )
    labels = labels[:, :128]
    sdf = get_sdf(labels) / labels.shape[1]
    delta = 0.05
    sdf = torch.clamp(sdf, -delta, delta) / delta

    from utils import plot_spectrogram, plot_labels

    fig, axs = plt.subplots(3, 1)
    plot_spectrogram(
        log_mel_specgram[0], title="log mel spectrogram", ax=axs[0], to_db=False
    )
    # show image on ax1
    plot_labels(labels, ax=axs[1])
    plot_labels(sdf, ax=axs[2])

    fig.tight_layout()
    plt.show()
