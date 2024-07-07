import os
import soundfile as sf
import torch
import torchaudio
from torchaudio.functional import mu_law_encoding
from torchaudio.transforms import Resample
from tqdm import tqdm


class RAVDESS(torch.utils.data.Dataset):
    def __init__(self, root_dir, orig_freq=48000, new_freq=16000, signal_length=32768):
        self.root_dir = root_dir
        self.signal_length = signal_length

        # resampler
        self.resample_freq = new_freq
        self.resampler = Resample(orig_freq=orig_freq, new_freq=self.resample_freq)

        self.data = []
        folder_list = os.listdir(self.root_dir)
        for folder in tqdm(folder_list, desc='Loading data', total=len(folder_list)):
            folder_path = os.path.join(self.root_dir, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                waveform, _ = torchaudio.load(file_path, normalize=True)  # data range [-1, 1]
                if waveform.shape[0] == 1:  # 5 samples are stereo, discard them
                    waveform = self.resampler(waveform)[:, self.resample_freq:]  # resample and cut 1st second

                    # pad with zeros if signal is shorter than signal_length
                    if waveform.shape[1] < self.signal_length:
                        pad_size = self.signal_length - waveform.shape[1]
                        waveform = torch.nn.functional.pad(waveform, (0, pad_size), mode='constant', value=0)

                    waveform = waveform[:, :self.signal_length]  # cut if signal is longer than signal_length
                    self.data.append((mu_law_encoding(waveform, 256) / 128) - 1)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dset = RAVDESS('../../data/RAVDESS')
