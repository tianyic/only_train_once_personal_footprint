import torch.nn as nn
import torch

class DemoNetConcatCase2(nn.Module):
    def __init__(self):
        super(DemoNetConcatCase2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_2 = nn.BatchNorm2d(128)
        self.bn_3 = nn.BatchNorm2d(128)
        self.bn_4 = nn.BatchNorm2d(768)
        self.relu = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(384, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.gemm1 = nn.Linear(in_features=768, out_features=128, bias=True)
        self.gemm2 = nn.Linear(in_features=128, out_features=10, bias=True)

    def forward(self, x):
        x_1 = self.relu(self.bn_1(self.conv1(x)))
        x_1_tmp = self.conv2(x_1)
        x_3 = self.conv3(x_1)
        x_2 = self.bn_2(x_1_tmp) + self.bn_3(x_1_tmp) + x_3
        x_4 = torch.cat([x_1, x_2], dim=1)
        x_5 = self.conv5(x_4)
        x_6 = self.conv6(torch.cat([x_5, x_2], dim=1))
        x_6 = self.bn_4(torch.cat([x_6, x_5], dim=1)) 
        x_7 = self.avg_pool(x_6)
        x_7 = x_7.view(x_7.size(0), -1)
        return self.gemm2(self.gemm1(x_7))