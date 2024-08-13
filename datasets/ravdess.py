import os
import torch
import torchaudio
from torchaudio.functional import mu_law_encoding
from torchaudio.transforms import Resample
from tqdm import tqdm

EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

class RAVDESS(torch.utils.data.Dataset):
    def __init__(self, root_dir, orig_freq=16000, new_freq=16000, signal_length=32768):
        self.root_dir = root_dir
        self.signal_length = signal_length
        self.orig_freq = orig_freq
        self.resample_freq = new_freq
        self.resampler = Resample(orig_freq=orig_freq, new_freq=self.resample_freq)

        self.data = []
        folder_list = os.listdir(self.root_dir)
        for folder in tqdm(folder_list, desc='Loading data', total=len(folder_list)):
            folder_path = os.path.join(self.root_dir, folder)
            for file in os.listdir(folder_path):
                cls = int(file.split('-')[2]) - 1  # class label

                file_path = os.path.join(folder_path, file)
                waveform, _ = torchaudio.load(file_path, normalize=True)  # data range [-1, 1]

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
                    self.data.append(((mu_law_encoding(waveform, 256) / 128) - 1, cls))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dset = RAVDESS('../../data/RAVDESS')
