import torch
import torch.nn as nn

# Define a simple 2D CNN model
class Simple2DCNN(nn.Module):
    def __init__(self):
        super(Simple2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # Adjust the size of the fully connected layer based on the output size of conv layers
        self.fc1 = nn.Linear(16 * 75 * 75, 128)  # Assuming input size is (300x300), downsampled by pooling twice
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 75 * 75)  # Flatten for the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x