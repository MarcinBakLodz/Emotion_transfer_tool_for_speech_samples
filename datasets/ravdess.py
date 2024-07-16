import os
import soundfile as sf
import torch
import torchaudio
from torchaudio.functional import mu_law_encoding
from torchaudio.transforms import Resample
from tqdm import tqdm


class RAVDESS(torch.utils.data.Dataset):
    def __init__(self, root_dir, orig_freq=16000, new_freq=16000, signal_length=32768):
        self.root_dir = root_dir
        self.signal_length = signal_length
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.resampler = Resample(orig_freq=self.orig_freq, new_freq=self.new_freq)

        self.data = []
        folder_list = os.listdir(self.root_dir)
        for folder in tqdm(folder_list, desc='Loading data', total=len(folder_list)):
            folder_path = os.path.join(self.root_dir, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                waveform, sample_rate = torchaudio.load(file_path, normalize=True)  # data range [-1, 1]

                if sample_rate != self.orig_freq:
                    raise ValueError(f"Unexpected sample rate {sample_rate}. Expected {self.orig_freq}")

                if waveform.shape[0] == 1:  # discard stereo samples
                    # Cut half a second from the start
                    start_trim = int(0.8 * self.orig_freq)
                    if waveform.shape[1] > start_trim:
                        waveform = waveform[:, start_trim:]

                    waveform = self.resampler(waveform)  # resample

                    # pad with zeros if signal is shorter than signal_length
                    if waveform.shape[1] < self.signal_length:
                        pad_size = self.signal_length - waveform.shape[1]
                        waveform = torch.nn.functional.pad(waveform, (0, pad_size), mode='constant', value=0)

                    waveform = waveform[:, :self.signal_length]  # cut if signal is longer than signal_length
                    self.data.append((mu_law_encoding(waveform, 256) / 128) - 1)
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    dset = RAVDESS('../../data/RAVDESS')
