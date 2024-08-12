import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchvision.transforms import Compose

class cooccurencePathDiscriminator(torch.nn.Module):
    def __init__(self, in_dim, h_dim = 0):
        self.in_dim=in_dim
        self.h_dim=h_dim
        super(cooccurencePathDiscriminator, self).__init__()
        self.f1 = torch.nn.Linear(in_dim,1)
    
    def forward(self, input):
        return self.f1(input)
    
if __name__ == "__main__":
    sample = torch.randn(size=(1, 32768))
    spec = cooccurencePathDiscriminator(in_dim=32768)
    y = spec(sample)
        