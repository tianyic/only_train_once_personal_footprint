import torch.nn as nn
import torch

class DemoNetConvtransposeInCase2(nn.Module):
    def __init__(self):
        super(DemoNetConvtransposeInCase2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in_1 = nn.InstanceNorm2d(64, affine=True)
        self.in_2 = nn.InstanceNorm2d(64, affine=True)
        self.leakyrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in_3 = nn.InstanceNorm2d(128, affine=True)
        self.in_6 = nn.InstanceNorm2d(512, affine=True)

        self.in_4 = nn.InstanceNorm2d(256, affine=True)
        self.in_5 = nn.InstanceNorm2d(256, affine=True)

        self.conv5 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv8 = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv9 = nn.ConvTranspose2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.gemm1 = nn.Linear(in_features=384, out_features=128, bias=True)
        self.gemm2 = nn.Linear(in_features=128, out_features=10, bias=True)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.leakyrelu(self.in_1(x_1)) 
        x_3 = self.leakyrelu(self.in_2(x_1))
        x_2 = self.conv3(x_2)        
        x_3 = self.conv4(self.leakyrelu(self.in_3(self.conv2(x_3))))
        x = torch.cat([x_2, x_3], dim=1)
        x = self.leakyrelu(self.in_6(x))

        x = self.leakyrelu(torch.cat([self.in_4(self.conv5(x)), self.in_5(self.conv6(x))], dim=1))
        x = self.leakyrelu(self.conv7(x))
        x = self.leakyrelu(self.conv8(x)) + self.leakyrelu(self.conv9(x))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.gemm2(self.gemm1(x))