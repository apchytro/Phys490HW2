import torch
import torch.nn as nn

# Class Net from nn_gen.py
# Handles the neural net using pytorch 

class Net(nn.Module):

    # Initialize the net 
    def __init__(self):
        super(Net, self).__init__()
        # Define convolution layers
        self.cnn_layers = nn.Sequential(
            # Convolution layer 1 output to 10 output
            nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
            # Batch Normalize
            nn.BatchNorm2d(10),
            # Activation ReLU
            nn.ReLU(inplace=True),
            # Max Pool to 7 * 7 pixels
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Convolution 10 to 10 
            nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
            # Batch Norm
            nn.BatchNorm2d(10),
            # ReLU
            nn.ReLU(inplace=True)
        )
        
        # Define linear layers
        self.linear_layers = nn.Sequential(
            # Fully connected layer z * x * y input to 5 output
            nn.Linear(10 * 7 * 7, 5)
        )
    
    # Forward step method
    def forward(self, x):
        # Apply layers
        f_x = self.cnn_layers(x)
        f_x = f_x.view(f_x.size(0), -1)
        f_x = self.linear_layers(f_x)
        return f_x
    
    # Back propagation method
    def backprop(self, data, loss, epoch, optimizer):
        # Train model
        self.train()
        # Initialize inputs and targets
        inputs= torch.from_numpy(data.x_train)
        targets= torch.from_numpy(data.y_train)
        # Check if GPU available
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        # Zero gradient optimizer
        optimizer.zero_grad()
        # Calculate outputs
        outputs= self(inputs)
        # Calculate losses
        obj_val= loss(outputs,targets.long())
        obj_val.backward()
        optimizer.step()
        return obj_val.item()
    
    # Method for test data
    def test(self, data, loss, epoch):
        # Initialize inputs and targets
        inputs= torch.from_numpy(data.x_test)
        targets= torch.from_numpy(data.y_test)
        # Check if GPU is available
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        # Calculate outputs
        outputs= self(inputs)
        # Calculate losses
        cross_val= loss(outputs,targets.long())
        return cross_val.item()
