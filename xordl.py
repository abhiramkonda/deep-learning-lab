import torch
import torch.nn as nn

# Define the neural network architecture
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.layer1 = nn.Linear(2, 2)  # Input layer to hidden layer
        self.activation = nn.Sigmoid()
        self.layer2 = nn.Linear(2, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return x

# Instantiate the model
model = XORModel()

# Display the model architecture and parameters
print(model)
print("\nNumber of learnable parameters:", sum(p.numel() for p in model.parameters()))
