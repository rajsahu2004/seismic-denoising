import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SeismicDataset
from model import SimpleCNN3D
from tqdm.auto import tqdm
from torch.amp import GradScaler, autocast
torch.cuda.empty_cache()


# Hyperparameters and settings
batch_size = 1  # Adjust according to your GPU memory
num_epochs = 10
learning_rate = 1e-3
data_dir = 'data'
torch.cuda.empty_cache()

device = torch.device('cpu')
device_type = 'cpu'

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

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    epoch_loss = 0.0
    num_batches = len(train_dataloader)

    # tqdm progress bar for batches
    batch_progress = tqdm(enumerate(train_dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}', total=num_batches)

    for i, (x, y) in batch_progress:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Automatic Mixed Precision (AMP) context
        with autocast(device_type=device_type):
            outputs = model(x)
            loss = criterion(outputs, y)

        # Backpropagation with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        epoch_loss += loss.item()

        # Update tqdm description for each batch
        batch_progress.set_postfix({'Batch Loss': loss.item(), 'Running Loss': running_loss / (i + 1)})

    # Average loss for the epoch
    avg_epoch_loss = epoch_loss / num_batches
    print(f'Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}')
    
    # Update learning rate
    scheduler.step()

    # Save model checkpoint
    torch.save(model.state_dict(), f'checkpoints/autoencoder_epoch_{epoch + 1}.pth')

print('Training complete')