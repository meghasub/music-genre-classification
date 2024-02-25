from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch
import os

# ANNOTATIONS_FILE = 'C:/Users/megha/Desktop/PythonProjects/Datasets/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv'
# AUDIO_DIR = 'C:/Users/megha/Desktop/PythonProjects/Datasets/UrbanSound8K/UrbanSound8K/audio'
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
import argparse


class UrbanSoundDataset(Dataset):

    def __init__(
        self,
        annotations_file,
        audio_dir,
        transformation,
        target_sample_rate,
        num_samples,
        device,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    # custom dataset should implement this function
    # we want to return the number of samples in the dataset
    def __len__(self):
        return len(self.annotations)

    # custom dataset should implement this function
    # what is getitem used for a_list[1] -> a_list.__getitem__(1)
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)

        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self._cut_if_necessary(signal)

        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples) -> (1, 50000) -> (1, 22050)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # Example of padding : [1,1,1] -> [1,1,1,0,0]
            num_missing_samples = self.num_samples - length_signal
            # In this case, the signal is (1, num_samples)
            # 0 specifies number of elements padded to the left of the LAST DIMENSION
            # num_missing_samples specifies number of elements padded to the right of the LAST DIMENSION
            # If we were to pad to second last dimension and so on.., then we would specify like this :
            # last_dim_padding = (nLSamples_Last, nRSamples_Last, nLSamples_SecondLast, nRSamples_SecondLast)
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(
                self.device
            )
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        audio_sample_path = os.path.join(
            self.audio_dir, fold, self.annotations.iloc[index, 0]
        )
        return audio_sample_path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-annotations_file", required=True)
    parser.add_argument("-audio_dir", required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f" Using device : {device}")
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    usd = UrbanSoundDataset(
        args.annotations_file,
        args.audio_dir,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
    )

    print(f"There are {len(usd)} samples in the dataset")
    signal, label = usd[1]
    print(type(signal), signal.shape)
    print(label)
