import torch.nn as nn
import torch

class DepthConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=1, groups=None):
        super(DepthConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=dilation, groups=1)
    
    def forward(self, x):
        return self.conv2(self.conv1(x))
    
normalizations = {
    'bn': nn.BatchNorm2d,
    'in': nn.InstanceNorm2d,
}

class DemoNetGroupConvCase1(nn.Module):
    def __init__(self, norm_type='in', affine=True, bias=True):
        super(DemoNetGroupConvCase1, self).__init__()
        self.conv_1 = DepthConv(6, 48, kernel_size=(5,5), stride=(2,2), padding=(2,2))
        self.in_1 = normalizations[norm_type](48, affine=affine)

        self.leakyrelu = nn.LeakyReLU()

        self.conv_2 = DepthConv(48, 96, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.in_2 = normalizations[norm_type](96, affine=affine)

        self.conv_3 = DepthConv(96, 192, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.in_3 = normalizations[norm_type](192, affine=affine)

        self.conv_4 = DepthConv(192, 384, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.in_4 = normalizations[norm_type](384, affine=affine)

        self.conv_5 = DepthConv(384, 384, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.in_5 = normalizations[norm_type](384, affine=affine)
    
        self.conv_6 = DepthConv(832, 1536, kernel_size=(2,2), stride=(1,1), padding=(1,1), dilation=2)

        self.convt_7 = nn.ConvTranspose2d(384, 384, kernel_size=(3, 3), padding=1, output_padding=1, stride=2)
        self.in_7 = normalizations[norm_type](384, affine=affine)

        self.convt_8 = nn.ConvTranspose2d(768, 192, kernel_size=(3, 3), padding=1, output_padding=1, stride=2)
        self.in_8 = normalizations[norm_type](192, affine=affine)

        self.convt_9 = nn.ConvTranspose2d(384, 96, kernel_size=(3, 3), padding=1, output_padding=1, stride=2)
        self.in_9 = normalizations[norm_type](96, affine=affine)

        self.convt_10 = nn.ConvTranspose2d(192, 48, kernel_size=(3, 3), padding=1, output_padding=1, stride=2)
        self.in_10 = normalizations[norm_type](48, affine=affine)

        self.conv_11 = nn.ConvTranspose2d(96, 48, kernel_size=(3, 3), padding=1, output_padding=1, stride=2)
        self.in_11 = normalizations[norm_type](48, affine=affine)

        self.conv_12 = nn.Conv2d(48, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.gemm1 = nn.Linear(in_features=96, out_features=48, bias=True)
        self.gemm2 = nn.Linear(in_features=48, out_features=10, bias=True)

        self.in_debug = normalizations[norm_type](384, affine=affine)

    def forward(self, x_1, x_2, x_3, x_4):
        x = torch.cat([x_1, x_2], dim=1)
        x = self.leakyrelu(self.in_1(self.conv_1(x)))

        x_down_1 = self.leakyrelu(self.in_2(self.conv_2(x)))
        x_down_2 = self.leakyrelu(self.in_3(self.conv_3(x_down_1)))
        x_down_3 = self.leakyrelu(self.in_4(self.conv_4(x_down_2)))
        x_down_4 = self.leakyrelu(self.in_5(self.conv_5(x_down_3)))

        x_down_4 = torch.cat([x_4, x_down_4, x_3], dim=1)
        x_down_4 = self.conv_6(x_down_4)
        x_down_4_up, x_down_4_out = x_down_4[:, :384, ...], x_down_4[:, 384:, ...]

        # print(x_down_4)
        # return x_down_4    
        # x_down_4_up, x_down_4_out = x_down_4, x_down_4
        
        x_up_1 = self.in_7(self.convt_7(x_down_4_up))

        # return x_up_1
    
        x_up_1 = torch.cat([x_up_1, x_down_3], dim=1)
        
        x_up_2 = self.in_8(self.convt_8(x_up_1))
        x_up_2 = torch.cat([x_up_2, x_down_2], dim=1)
        x_up_3 = self.in_9(self.convt_9(x_up_2))
        x_up_3 = torch.cat([x_up_3, x_down_1], dim=1)
        x_up_4 = self.in_10(self.convt_10(x_up_3))
        x_up_4 = torch.cat([x_up_4, x], dim=1)
        x_up_5 = self.in_11(self.conv_11(x_up_4))

        x_out = self.conv_12(x_up_5)
        x_out = self.avg_pool(x_out)
        x_out = x_out.view(x_out.size(0), -1)
        
        return self.gemm2(self.gemm1(x_out)), x_down_4_out
        # return self.gemm2(self.gemm1(x_out))
# net = DemoNetGroupConvCase1()

# dummy_input_1 = torch.rand(1, 3, 512, 512)
# dummy_input_2 = torch.rand(1, 3, 512, 512)
# dummy_input_3 = torch.rand(1, 384, 16, 16)
# dummy_input_4 = torch.rand(1, 64, 16, 16)

# outs = net(dummy_input_1, dummy_input_2, dummy_input_3, dummy_input_4)
# print(outs[0].shape, outs[1].shape)
