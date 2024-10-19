import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SeismicDataset
from model import BaseModel
from tqdm.auto import tqdm
import numpy as np

# Initialize the dataset and dataloader
test_dataset = SeismicDataset(data_dir='data', train=False)  # Change 'train=False' for testing
batch_size = 1  # Adjust according to memory capacity
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = BaseModel(in_channels=1259, out_channels=1259)

# Load the trained model weights
model.load_state_dict(torch.load('model.pth'))

# Move model to device (GPU if available)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Testing loop
model.eval()  # Set the model to evaluation mode
running_loss = 0.0
total_psnr = 0.0  # To calculate average PSNR
criterion = nn.MSELoss()

with torch.no_grad():  # No gradient calculation during testing
    for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
        inputs, labels = inputs.float().to(device), labels.float().to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Optionally calculate PSNR for evaluation
        psnr_value = 10 * np.log10(1 / loss.item())  # Assuming the range of input is [0, 1]
        total_psnr += psnr_value
        
        if i % 5 == 0:
            print(f'Step [{i}/{len(test_dataloader)}], Loss: {loss.item():.4f}, PSNR: {psnr_value:.4f}')

avg_loss = running_loss / len(test_dataloader)
avg_psnr = total_psnr / len(test_dataloader)
print(f'Average Loss: {avg_loss:.4f}, Average PSNR: {avg_psnr:.4f}')

print('Testing complete!')