import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization

class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = F.max_pool2d((F.relu(self.conv1(x))), 2)
        x = F.max_pool2d((F.relu(self.conv2(x))), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NetQuant(nn.Module):
    def __init__(self, num_channels=1):
        super(NetQuant, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(5*5*40, 10)

        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(-1, 5*5*40)
        x = self.fc(x)
        x = self.dequant(x)
        return x
    

class BiConv(nn.Module):
    def __init__(self, num_channels=1):
        super(BiConv, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(40)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(40, 40, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(40)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(128*128*40, 10)

        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.bn1(self.conv1(x))
        x = self.relu1(x)
        x = self.bn2(self.conv2(x))
        x = self.relu2(x)
        x = x.reshape(1, 128*128*40)
        x = self.fc(x)
        x = self.dequant(x)
        return x