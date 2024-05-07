import torch
import torch.optim as optim

class Encoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, latent_dim):
        super().__init__()
        self.conv_stack = torch.nn.Sequential(
            # 6 strided convolutional layers with stride 2 and window size 4
            torch.nn.Conv1d(in_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            # the latents consist of one feature map and the discrete space is 512-dimensional
            torch.nn.Conv1d(h_dim, latent_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.conv_stack(x)


class Decoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()

        self.inverse_conv_stack = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(h_dim, 1, kernel_size=6, stride=2, padding=1))

    def forward(self, x):
        return self.inverse_conv_stack(x)


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, inputs, epochs=10):
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = self.criterion(outputs, inputs)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

class AutoEncoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, h_dim, latent_dim)
        self.decoder = Decoder(latent_dim, h_dim)

    def forward(self, x):
        print("raw: ", x.shape)
        x = self.encoder(x)
        print("enc: ", x.shape)
        x = self.decoder(x)
        print("dec: ", x.shape)
        return x

# Example usage:
if __name__ == "__main__":
    # Define model parameters
    in_dim = 1
    h_dim = 64
    latent_dim = 512

    # Instantiate the model
    autoencoder = AutoEncoder(in_dim, h_dim, latent_dim)

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Define the optimizer
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Instantiate the trainer
    trainer = Trainer(autoencoder, criterion, optimizer)

    # Generate some dummy input and target data
    input_data = torch.randn(32, in_dim, 400)  # Batch size 32, input dimension 1, sequence length 100

    # Train the autoencoder
    trainer.train(input_data)