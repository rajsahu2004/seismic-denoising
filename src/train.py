import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SeismicDataset
from model import Simple2DCNN

# Initialize the dataset and dataloader
dataset = SeismicDataset(data_dir='data', train=True)
batch_size = 1  # Adjust according to memory capacity
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = Simple2DCNN()
criterion = nn.MSELoss()  # Assuming you're doing regression on seismic data
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
num_epochs = 10  # Adjust for faster runs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        # Move inputs and labels to the correct device and add the channel dimension
        inputs, labels = inputs.float().to(device), labels.float().to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate the loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()

        # Optional: Gradient Clipping (to avoid exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step the optimizer
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        
        # Print the loss every 10 batches
        if i % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}')
    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')
print('Training complete!')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')