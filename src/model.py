import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1), # Output: (32, 1259, 300, 300)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),     # Output: (32, 629, 150, 150)
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), # Output: (64, 629, 150, 150)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),     # Output: (64, 314, 75, 75)
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1), # Output: (128, 314, 75, 75)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)      # Output: (128, 157, 37, 37)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),  # Output: (64, 314, 75, 75)
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),   # Output: (32, 629, 150, 150)
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=2, stride=2),    # Output: (1, 1259, 300, 300)
            nn.Sigmoid() # Sigmoid activation to ensure output is in range [0, 1]
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x