import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import torch

def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module
        
class DemonetBatchnormPruning(nn.Module):
    def __init__(self, input_nc, base_nc, max_nc, encoder_layers, decoder_layers, nonlinearity, use_spect, size=256):
        super(DemonetBatchnormPruning, self).__init__()

        if size == 512:
            self.input_layer = nn.Sequential(nn.Conv2d(input_nc, base_nc, kernel_size=7, stride=2, padding=3),
                                             nn.Conv2d(base_nc, base_nc, kernel_size=7, stride=1, padding=3))
        elif size == 256:
            self.input_layer = nn.Conv2d(input_nc, base_nc, kernel_size=7, stride=1, padding=3)
        elif size == 64:
            self.input_layer = nn.Conv2d(input_nc, base_nc, kernel_size=3, stride=1, padding=1)
        else:
            raise Exception('Input layer for the size is not defined: ', size)

        for i in range(encoder_layers):
            in_channels = min(base_nc * 2**i, max_nc)
            out_channels = min(base_nc * 2**(i+1), max_nc)
            model = ResBlock(in_channels, out_channels, out_channels, use_transpose=False,
                                   nonlinearity=nonlinearity, use_spect=use_spect)
            setattr(self, 'encoder' + str(i), model)

        for i in range(encoder_layers - decoder_layers, encoder_layers)[::-1]:
            in_channels = min(base_nc * (2 ** (i + 1)), max_nc)
            in_channels = in_channels * 2 if i != (encoder_layers - 1) else in_channels
            out_channels = min(base_nc * (2 ** i), max_nc)
            model = ResBlock(in_channels, out_channels, out_channels, use_transpose=True,
                                   nonlinearity=nonlinearity, use_spect=use_spect)
            setattr(self, 'decoder' + str(i), model)

        self.output_nc = out_channels * 2
        self.output_layer = nn.Conv2d(self.output_nc, self.output_nc, kernel_size=3, stride=1, padding=1)

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

    def forward(self, x):
        x = torch.cat(x, dim=1)
        out = self.input_layer(x)
        out_list = [out]
        for i in range(self.encoder_layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            out_list.append(out)

        out = out_list.pop()
        for i in range(self.encoder_layers-self.decoder_layers, self.encoder_layers)[::-1]:
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
            out = torch.cat([out, out_list.pop()], 1)
            
        out = self.output_layer(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, input_nc, output_nc, hidden_nc, use_transpose=True, nonlinearity=nn.LeakyReLU(),
                 use_spect=False):
        super(ResBlock, self).__init__()
        # Attributes
        self.actvn = nonlinearity
        hidden_nc = min(input_nc, output_nc) if hidden_nc is None else hidden_nc

        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        if use_transpose:
            kwargs_up = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        else:
            kwargs_up = {'kernel_size': 3, 'stride': 2, 'padding': 1}

        # create conv layers
        self.conv_0 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, **kwargs_fine), use_spect)
        if use_transpose:
            self.conv_1 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, **kwargs_up), use_spect)
            self.conv_s = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, **kwargs_up), use_spect)
        else:
            self.conv_1 = nn.Sequential(spectral_norm(nn.Conv2d(hidden_nc, output_nc, **kwargs_up), use_spect))#,
                                        # nn.Upsample(scale_factor=2))
            self.conv_s = nn.Sequential(spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs_up), use_spect))#,
                                        # nn.Upsample(scale_factor=2))
        # # define normalization layers
        self.norm_0 = nn.BatchNorm2d(input_nc)
        self.norm_1 = nn.BatchNorm2d(hidden_nc)
        self.norm_s = nn.BatchNorm2d(input_nc)

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv_0(self.actvn(self.norm_0(x)))
        dx = self.conv_1(self.actvn(self.norm_1(dx)))
        out = x_s + dx
        return out

    def shortcut(self, x):
        x_s = self.conv_s(self.actvn(self.norm_s(x)))
        return x_s


if __name__=="__main__":

    net = DemonetBatchnormPruning(13,32,256,5,3,nn.LeakyReLU(),False,256)
    input = torch.randn(1,13,256,256)
    out = net(input)
    print(out.shape)