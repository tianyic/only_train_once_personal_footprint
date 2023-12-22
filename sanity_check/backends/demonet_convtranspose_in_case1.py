import torch.nn as nn
import torch

AFFINE=True

class DemoNetConvtransposeInCase1(nn.Module):
    def __init__(self):
        super(DemoNetConvtransposeInCase1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_1 = nn.InstanceNorm2d(64, affine=AFFINE)
        self.bn_2 = nn.InstanceNorm2d(64, affine=AFFINE)
        self.leakyrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_3 = nn.InstanceNorm2d(128, affine=AFFINE)

        self.conv6 = nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)

        self.bn_6 = nn.InstanceNorm2d(512, affine=AFFINE)
        
        self.conv8 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv9 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.in_1 = nn.InstanceNorm2d(256, affine=AFFINE)
        self.in_2 = nn.InstanceNorm2d(256, affine=AFFINE)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.gemm1 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.gemm2 = nn.Linear(in_features=128, out_features=10, bias=True)
        
        
    def forward(self, x, debug=False):
        x_1 = self.conv1(x)
        x_2 = self.leakyrelu(self.bn_1(x_1)) 
        x_3 = self.leakyrelu(self.bn_2(x_1))
        x_2 = self.conv3(x_2)
        x_3 = self.conv4(self.leakyrelu(self.bn_3(self.conv2(x_3))))
        x = x_2 + x_3
        x = self.leakyrelu(self.bn_6(self.conv6(x)))
        x = self.in_1(self.conv8(x)) + self.in_2(self.conv9(x)) 

        x = self.avg_pool(self.conv10(x))
        x = x.view(x.size(0), -1)
        
        return self.gemm2(self.gemm1(x))