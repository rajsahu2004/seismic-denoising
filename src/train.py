import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

# ---------------- Helper Functions ---------------- #
def load_and_preprocess_data(img_path, noise_path):
    img = np.load(img_path, allow_pickle=True)
    noise = np.load(noise_path, allow_pickle=True)
    
    img = check_shape(img)
    noise = check_shape(noise)
    img = rescale_volume(img, 1, 99)
    noise = rescale_volume(noise, 1, 99)
    
    img = img.astype(np.float32)
    noise = noise.astype(np.float32)
    
    return img, noise

def check_shape(npy_file):
    if npy_file.shape != (1259, 300, 300):
        npy_file = npy_file.T
    return npy_file

def rescale_volume(seismic, low=0, high=100):
    minval = np.percentile(seismic, low)
    maxval = np.percentile(seismic, high)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255
    return seismic

# ---------------- Dataset Creation ---------------- #
class SeismicDataset(Dataset):
    def __init__(self, img_paths, noise_paths):
        self.img_paths = img_paths
        self.noise_paths = noise_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img, noise = load_and_preprocess_data(self.img_paths[idx], self.noise_paths[idx])
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        noise = torch.tensor(noise, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        return img, noise

def create_dataset(pairs, batch_size=8):
    img_paths, noise_paths = zip(*pairs)
    dataset = SeismicDataset(img_paths, noise_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dataloader
# ---------------- Model Definition ---------------- #
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ---------------- Training Function ---------------- #
def train_model(train_loader, val_loader, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for img, noise in train_loader:
            img, noise = img.to(device), noise.to(device)
            
            optimizer.zero_grad()
            outputs = model(noise)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * img.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, noise in val_loader:
                img, noise = img.to(device), noise.to(device)
                outputs = model(noise)
                loss = criterion(outputs, img)
                val_loss += loss.item() * img.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'seismic_denoising_model.pth')

    # Plot loss curves
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig('training_loss_curve.png')


if __name__ == "__main__":
    file_pairs = [
        ('data/training_data/77695365/seismicCubes_RFC_fullstack_2023.77695365.npy', 'data/training_data/77695365/seismic_w_noise_vol_77695365.npy'),
        ('data/training_data/76135802/seismicCubes_RFC_fullstack_2023.76135802.npy', 'data/training_data/76135802/seismic_w_noise_vol_76135802.npy'),
        # Add all your file pairs here...
    ]

    # Split data into training and validation sets
    train_pairs, val_pairs = random_split(file_pairs, [int(0.8 * len(file_pairs)), int(0.2 * len(file_pairs))])
    
    # Create datasets
    train_loader = create_dataset(train_pairs, batch_size=8)
    val_loader = create_dataset(val_pairs, batch_size=8)
    
    for img, noise in train_loader:
        print("Image shape:", img.shape)
        print("Noise shape:", noise.shape)
        break

    # Train the model
    train_model(train_loader, val_loader, epochs=50)