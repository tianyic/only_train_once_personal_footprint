import torch.nn as nn
import torch

class DemoNetWeightShareCase2(nn.Module):
    def __init__(self):
        super(DemoNetWeightShareCase2, self).__init__()
        self.conv1 = nn.Conv2d(3, 832, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(832, 416, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(832, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(1536, 416, kernel_size=(1, 1), stride=(1, 1))
        self.conv5 = nn.Conv2d(832, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.num_layers = 4

    def forward(self, x):
        x = self.conv1(x)
        skip_x = self.conv2(x)
        for i in range(self.num_layers):
            x = torch.cat([self.conv4(self.conv3(x)), skip_x], dim=1)
        return self.conv5(x)
