import torch


# source: https://github.com/MishaLaskin/vqvae/blob/master/main.py
class VectorQuantizer(torch.nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta=0.25):
        super().__init__()
        self.device = 'cpu'

        self.n_e = n_e  # K in article
        self.e_dim = e_dim  # D in article
        # technically D HAS TO BE equal C (number of filters in last layer of the encoder)
        # but typically they use 1x1 convolution to fix dimensions

        self.beta = beta

        self.embedding = torch.nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)  # codebook initialization

    def set_device(self, device):
        self.device = device

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 1).contiguous()

        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # find the closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        """
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        ↑↑↑↑
        This creates a one-hot vector for each element in the flattened input z. 
        It does this by setting the value of the index corresponding to the closest embedding to 1, and the rest to 0.
        For example, if the min_encoding_indices are [2, 5, 1],
        then the min_encodings will be [[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0]].
        This way, the min_encodings can be used to look up the embedding vectors by matrix multiplication.
        """
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight)
        z_q = z_q.view(z.shape)

        # compute loss for embedding
        """
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        ↑↑↑↑
        This line is calculating the loss for the Vector Quantizer layer. The loss is composed of two terms:

        torch.mean((z_q.detach()-z)**2): This is the commitment loss, which measures the difference between
                                         the output of the encoder network z and the quantized output z_q.
                                         The detach() function is used to stop the gradients from being backpropagated 
                                         through z_q during this part of the loss calculation.
                                         The mean squared difference is then calculated to get a single scalar value for the loss.

        self.beta * torch.mean((z_q - z.detach()) ** 2): This is the codebook loss, which encourages the quantized output z_q
                                                         to be close to the output of the encoder network z.
                                                         The detach() function is used to stop the gradients from being
                                                         backpropagated through z during this part of the loss calculation.
                                                         The mean squared difference is then calculated and multiplied by
                                                         a hyperparameter beta to control the weighting of this term in the loss.

        The total loss is the sum of these two terms. It’s used to train the Vector Quantizer layer to produce quantized outputs
        that are close to the outputs of the encoder network, while also ensuring that each vector in the embedding space
        gets updated.
        """
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients using Straight-Through Estimator
        z_q = z + (z_q - z).detach()

        # perplexity
        """
        Perplexity is used as a measure of how well the discrete latent variables (the encodings) are being used.
        If the perplexity is low, it means that only a few latent variables are being used frequently,
        while the others are rarely used.
        If the perplexity is high, it means that the latent variables are being used more uniformly.
        """
        e_mean = torch.mean(min_encodings, dim=0)  # The mean of the encodings over the batch dimension.
        # The result is a vector where each element is the average usage of
        # a particular encoding in the batch.
        """
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
                               ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        The expression inside the torch.exp() function is the entropy of the encoding distribution,
        which measures the uncertainty or randomness of the distribution.
        The 1e-10 is a small constant added for numerical stability, to avoid taking the logarithm of zero.
        The entropy is negated and exponentiated to calculate the perplexity.
        """
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 2, 1).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation=torch.nn.LeakyReLU(inplace=True), stride=2):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.activation = activation
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.downsample = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out += residual
        out = self.activation(out)
        return out


class UpsampleResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation=torch.nn.LeakyReLU(inplace=True), stride=2):
        super(UpsampleResidualBlock, self).__init__()
        self.upsample = torch.nn.Upsample(scale_factor=stride, mode='nearest')
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.upsample(x)
        residual = self.conv3(residual)
        out = self.conv1(x)
        out = self.activation(out)
        out = self.upsample(out)
        out = self.conv2(out)
        out += residual
        out = self.activation(out)
        return out


class Encoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, latent_dim):
        super().__init__()
        self.conv_stack = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, h_dim, kernel_size=7, padding=3),
            torch.nn.LeakyReLU(inplace=True),
            ResidualBlock(h_dim, h_dim),
            ResidualBlock(h_dim, h_dim),
            ResidualBlock(h_dim, h_dim),
            torch.nn.Conv1d(h_dim, latent_dim, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.conv_stack(x)


class Decoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()

        self.inverse_conv_stack = torch.nn.Sequential(
            UpsampleResidualBlock(in_dim, h_dim),
            UpsampleResidualBlock(h_dim, h_dim),
            UpsampleResidualBlock(h_dim, h_dim),
            UpsampleResidualBlock(h_dim, h_dim),
            torch.nn.Conv1d(h_dim, 1, kernel_size=7, padding=3),
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


class DualLatentEncoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, cont_latent_dim, vq_latent_dim, n_e, beta):
        super().__init__()

        self.conv_stack_1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, h_dim, kernel_size=7, padding=3),
            torch.nn.LeakyReLU(),
            ResidualBlock(h_dim, h_dim),
            ResidualBlock(h_dim, h_dim),
            ResidualBlock(h_dim, h_dim)
        )

        self.conv_stack_2 = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            ResidualBlock(h_dim, cont_latent_dim)
        )

        self.vector_quantization = torch.nn.Sequential(
            torch.nn.Conv1d(h_dim, vq_latent_dim, kernel_size=3, stride=2, padding=1),
            VectorQuantizer(n_e, vq_latent_dim, beta)
        )

    def forward(self, x):
        x = self.conv_stack_1(x)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(x)
        x = self.conv_stack_2(x)
        return x, z_q, embedding_loss, perplexity


class DualLatentDecoder(Decoder):
    def __init__(self, in_dim, h_dim):
        super().__init__(in_dim, h_dim)

    def forward(self, x, z_q):
        return self.inverse_conv_stack(torch.concatenate((x, z_q), dim=1))


# TODO: INTEGRATE AND TEST DISCRIMINATORS
# source: https://github.com/kaiidams/soundstream-pytorch/blob/main/soundstream.py
class WaveDiscriminator(torch.nn.Module):
    r"""MelGAN discriminator from https://arxiv.org/pdf/1910.06711.pdf
    """
    def __init__(self, resolution: int = 1, n_channels: int = 4) -> None:
        super().__init__()
        assert resolution >= 1

        if resolution == 1:
            self.avg_pool = torch.nn.Identity()
        else:
            self.avg_pool = torch.nn.AvgPool1d(resolution * 2, stride=resolution)

        self.activation = torch.nn.LeakyReLU(0.2, inplace=True)

        self.layers = torch.nn.ModuleList([
            torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(1, n_channels, kernel_size=15, padding=7)),
            torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(n_channels, 4 * n_channels, kernel_size=41, stride=4, padding=20, groups=4)),
            torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(4 * n_channels, 8 * n_channels, kernel_size=41, stride=4, padding=20, groups=8)),
            torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(8 * n_channels, 16 * n_channels, kernel_size=41, stride=4, padding=20, groups=16)),
            torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(16 * n_channels, 32 * n_channels, kernel_size=41, stride=4, padding=20, groups=32)),
            torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(32 * n_channels, 32 * n_channels, kernel_size=5, padding=2)),
            torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(32 * n_channels, 1, kernel_size=3, padding=1)),
        ])

    def forward(self, x):
        x = self.avg_pool(x)
        feats = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
            x = self.activation(x)
        feats.append(self.layers[-1](x))
        return feats


class ResNet2d(torch.nn.Module):
    def __init__(self, n_channels, factor, stride):
        # https://arxiv.org/pdf/2005.00341.pdf the original paper uses layer normalization, but here we use batch normalization.
        super().__init__()

        self.conv0 = torch.nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), padding='same')
        self.bn0 = torch.nn.BatchNorm2d(n_channels)
        self.conv1 = torch.nn.Conv2d(n_channels, factor * n_channels, kernel_size=(stride[0] + 2, stride[1] + 2), stride=stride)
        self.bn1 = torch.nn.BatchNorm2d(factor * n_channels)
        self.conv2 = torch.nn.Conv2d(n_channels, factor * n_channels, kernel_size=1, stride=stride)
        self.bn2 = torch.nn.BatchNorm2d(factor * n_channels)
        self.pad = torch.nn.ReflectionPad2d([(stride[1] + 1) // 2, (stride[1] + 2) // 2, (stride[0] + 1) // 2, (stride[0] + 2) // 2])
        self.activation = torch.nn.LeakyReLU(0.3)

    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = self.activation(x)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)

        # shortcut
        y = self.conv2(input)
        y = self.bn2(y)

        x += y
        x = self.activation(x)
        return x


class STFTDiscriminator(torch.nn.Module):
    r"""STFT-based discriminator from https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(
        self, n_fft: int = 1024, hop_length: int = 256,
        n_channels: int = 32
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n = n_fft // 2 + 1
        for _ in range(6):
            n = (n - 1) // 2 + 1

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, n_channels, kernel_size=7, padding='same'),
            torch.nn.LeakyReLU(0.3, inplace=True),
            ResNet2d(n_channels, 2, stride=(2, 1)),
            ResNet2d(2 * n_channels, 2, stride=(2, 2)),
            ResNet2d(4 * n_channels, 1, stride=(2, 1)),
            ResNet2d(4 * n_channels, 2, stride=(2, 2)),
            ResNet2d(8 * n_channels, 1, stride=(2, 1)),
            ResNet2d(8 * n_channels, 2, stride=(2, 2)),
            torch.nn.Conv2d(16 * n_channels, 1, kernel_size=(n, 1))
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[1] == 1
        # input: [batch, channel, sequence]
        x = torch.squeeze(input, 1).to(torch.float32)  # torch.stft() doesn't accept float16
        x = torch.stft(x, self.n_fft, self.hop_length, normalized=True, onesided=True, return_complex=True)
        x = torch.abs(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.layers(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    sample = torch.randn(size=(1, 1, 32768))

    # enc = DualLatentEncoder(in_dim=1, h_dim=256, cont_latent_dim=16, n_e=512, vq_latent_dim=16, beta=0.25, verbose=False)
    enc = Encoder(in_dim=1, h_dim=256, latent_dim=1)
    summ = summary(enc, input_data=sample)

    # disc = WaveDiscriminator(resolution=4, n_channels=4)
    # summ = summary(disc, input_data=sample)
    # out = disc(sample.to('cuda'))
    # print(out[-1].size())

    # latent = torch.randn(size=(1, 2048))
    # dec = Decoder(in_dim=1, h_dim=32)
    # summ2 = summary(dec, input_data=latent.to('cuda'), device='cuda')
