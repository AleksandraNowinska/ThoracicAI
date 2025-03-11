import torch
import torch.nn as nn


class Net(nn.Module):
    #27.31it/s -> 2.14it/s - performance after first committ (adding changes in net only)
    def __init__(self, n_classes: int) -> None:
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # First Convolution Block (Extracts basic features like edges)
            nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Second Convolution Block (Expanding to capture complexity)
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # Pooling applied here

            # Third Convolution Block (Starting to contract feature maps)
            nn.Conv2d(128, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Fourth Convolution Block (Final feature extraction before flattening)
            nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # Ensuring controlled downsampling

            nn.Dropout(p=0.4)  # Dropout added at the end to prevent overfitting
        )

        # Compute the size dynamically so we don’t have to manually set input size
        self.flattened_size = self._get_flattened_size()

        self.linear_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def _get_flattened_size(self) -> int:
        """Passes a dummy input through CNN layers to compute the output size"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 128)  # Single grayscale image
            output = self.cnn_layers(dummy_input)
            return output.view(1, -1).size(1)  # Compute flattened size


    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
