import torch
import torch.nn as nn

class SimpleCNN3D(nn.Module):
    def __init__(self):
        super(SimpleCNN3D, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 1200, 300, 300)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),       # Output: (32, 600, 150, 150)

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 600, 150, 150)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),       # Output: (64, 300, 75, 75)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0),   # Output: (32, 600, 150, 150)
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=2, stride=2, padding=0),    # Output: (1, 1200, 300, 300)
            nn.Sigmoid()  # Use Sigmoid to map output to [0, 1] range
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x