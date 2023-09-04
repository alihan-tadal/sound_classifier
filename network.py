from torch import nn

# from torchsummary import summary


class Network(nn.Module):
    # Summary: 4 conv block, flatten, linear, softmax
    def __init__(self):
        # Maymuni bir şekilde inşa et.
        # TODO: Constructor'a verilen parametrelerle daha dinamik ağ inşası sağla.
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,  # Usual for kernel size.
                stride=1,
                padding=2,  # Usual values.
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,  # Usual for kernel size.
                stride=1,
                padding=2,  # Usual values.
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,  # Usual for kernel size.
                stride=1,
                padding=2,  # Usual values.
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,  # Usual for kernel size.
                stride=1,
                padding=2,  # Usual values.
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 10)  # Equals flatten output.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    network = Network()
    print(network)  # Get summary of the model.
