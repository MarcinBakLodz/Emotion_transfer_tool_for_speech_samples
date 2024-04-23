import os
import soundfile as sf
import torch
import torchaudio
from torchaudio.datasets import VCTK_092
from torchaudio.functional import mu_law_encoding
from torchaudio.transforms import Resample
from tqdm import tqdm


class VCTK(torch.utils.data.Dataset):
    def __init__(self, root_dir, orig_freq=48000, new_freq=16000, segment_len=40960):
        # download VCTK and save into root_dir
        self.dataset = VCTK_092(root=root_dir, download=True)

        # resampler
        self.resample_freq = new_freq
        self.resampler = Resample(orig_freq=orig_freq, new_freq=self.resample_freq)

        self.segment_len = segment_len
        self.root_dir = os.path.join(root_dir, 'VCTK-Corpus-0.92')
        self.segments_dir = os.path.join(self.root_dir, 'segments')

        # preprocess if segments not saved
        if not os.path.exists(self.segments_dir):
            os.makedirs(self.segments_dir)
            self.preprocess()

        # Load all data into memory
        self.data = []
        for file in tqdm([f for f in os.listdir(self.segments_dir)], desc='Loading data'):
            waveform, _ = torchaudio.load(os.path.join(self.segments_dir, file), normalize=True)  # data range [-1, 1]
            # TODO check the influence of mu_law_encoding on the results
            self.data.append((mu_law_encoding(waveform, 256) / 128) - 1)  # refer to -> https://en.wikipedia.org/wiki/Μ-law_algorithm, https://pytorch.org/audio/main/generated/torchaudio.transforms.MuLawEncoding.html

    def preprocess(self):
        segment_index = 0
        for i in tqdm(range(len(self.dataset)), desc='Preprocessing data'):
            waveform, *_ = self.dataset[i]
            waveform = self.resampler(waveform)

            # If waveform is shorter than self.segment_len, pad it with zeros
            if waveform.shape[1] < self.segment_len:
                pad_size = self.segment_len - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_size), mode='constant', value=0)

            for start in range(0, waveform.shape[1] - self.segment_len + 1, self.segment_len):
                segment = waveform[:, start: start + self.segment_len]
                sf.write(os.path.join(self.segments_dir, f'{segment_index}.wav'), segment.t().numpy(), self.resample_freq)
                segment_index += 1

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
