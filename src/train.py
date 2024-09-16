import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SeismicDataset
from model import SimpleCNN3D
from tqdm.auto import tqdm
from torch.amp import GradScaler, autocast

# Hyperparameters and settings
batch_size = 1  # Adjust according to your GPU memory
num_epochs = 10
learning_rate = 1e-3
data_dir = 'data'
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# Initialize the dataset and dataloader
train_dataset = SeismicDataset(data_dir=data_dir, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=64)  # Adjust num_workers as needed
# Initialize the model, loss function, and optimizer
model = SimpleCNN3D().to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create directory for saving model checkpoints
os.makedirs('checkpoints', exist_ok=True)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Gradient scaler for mixed precision training
scaler = GradScaler()

for epoch in tqdm(range(num_epochs), desc='Epochs', total=num_epochs):
    model.train()
    running_loss = 0.0
    epoch_loss = 0.0
    num_batches = len(train_dataloader)
    
    for i, (x, y) in enumerate(train_dataloader):
        print(x.shape, y.shape)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            outputs = model(x)
            loss = criterion(outputs, x)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        epoch_loss += loss.item()

        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{num_batches}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    # Print average loss for the epoch
    avg_epoch_loss = epoch_loss / num_batches
    print(f'Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}')
    
    # Update learning rate
    scheduler.step()

    # Save model checkpoint
    torch.save(model.state_dict(), f'checkpoints/autoencoder_epoch_{epoch + 1}.pth')

print('Training complete')
