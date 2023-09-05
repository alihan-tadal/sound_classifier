import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset


class UrbanSoundDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        audio_dir,
        transformotion,
        target_sample_rate,
        num_samples,
        device,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformotion = transformotion.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        sample_path = self._get_sample_path(index)
        label = self._get_audio_label(index)
        signal, sample_rate = torchaudio.load(sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sample_rate)
        signal = self._mix_down_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self.transformotion(signal)
        return signal, label

    # Helpers
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing = self.num_samples - length_signal
            padding = (0, num_missing)  # 0 for left, num_missing for right padding.
            signal = torch.nn.functional.pad(signal, padding)
        return signal

    def _resample_if_necessary(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sample_rate
            )
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        # signal: (num_channels, samples) looks like (2, 16000).
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"  # Column 5: fold
        path = os.path.join(
            self.audio_dir, fold, self.annotations.iloc[index, 0]
        )  # Column 0: slice_file_name
        return path

    def _get_audio_label(self, index):
        return self.annotations.iloc[index, 6]


if "__name__" == "__main__":
    ANNOTATION_FILE = "../data/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "../data/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    usd = UrbanSoundDataset(
        ANNOTATION_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE, NUM_SAMPLES, device
    )

    