import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BaseModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),  # (512, H, W)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample (512, H//2, W//2)

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # (256, H//2, W//2)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample (256, H//4, W//4)
        )
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2),  # Upsample (512, H//2, W//2)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, out_channels, kernel_size=2, stride=2)  # Adjust output_padding
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
