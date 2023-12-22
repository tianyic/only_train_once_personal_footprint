import torch
import torch.nn as nn

class DemoNetInstanceNorm2DCase3(nn.Module):
    def __init__(self):
        super(DemoNetInstanceNorm2DCase3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.in_1 = nn.InstanceNorm2d(64, affine=True)
        self.in_2 = nn.InstanceNorm2d(128, affine=True)
        self.in_3 = nn.InstanceNorm2d(256, affine=True)
        self.in_4 = nn.InstanceNorm2d(512, affine=True)
        
        self.leakyrelu = nn.LeakyReLU()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.gemm1 = nn.Linear(in_features=512, out_features=128, bias=True)
        self.gemm2 = nn.Linear(in_features=128, out_features=10, bias=True)

    def forward(self, x):
        x = self.leakyrelu(self.in_1(self.conv1(x)))
        x = self.leakyrelu(self.in_2(self.conv2(x)))
        x = self.leakyrelu(self.in_3(self.conv3(x)))
        x = self.leakyrelu(self.in_4(self.conv4(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.gemm2(self.gemm1(x))