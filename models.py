# This code part is copied from 1Konny/WAE-pytorch.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import gin

import math
import padding


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class PadSame(nn.Module):
    def __init__(self, kernel_size, stride, dilation):
        super(PadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        return padding.pad_same(x, self.kernel_size, self.stride, self.dilation)

@gin.configurable
class WAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=64, nc=3, distribution = 'sphere', input_normalize_sym=False):
        super(WAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.distribution = distribution
        self.input_normalize_sym = input_normalize_sym
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*4*4)),                                 # B, 1024*4*4
            nn.Linear(1024*4*4, z_dim)                            # B, z_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024*8*8),                           # B, 1024*8*8
            View((-1, 1024, 8, 8)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)

        return x_recon, z

    def _encode(self, x):
        if self.distribution == "sphere":
            return F.normalize(self.encoder(x), dim=1, p=2)
        else:
            return self.encoder(x)

    def _decode(self, z):
        xd = self.decoder(z)
        if self.input_normalize_sym:
            return F.tanh(xd)
        else:
            return F.sigmoid(xd)


@gin.configurable
class DcganModel(nn.Module):
    """DCGAN style Encoder-Decoder architecture."""
    def __init__(self, z_dim=10, nc=3, input_dims=(32, 32, 3),
            distribution=gin.REQUIRED,
            input_normalize_sym=gin.REQUIRED,
            e_num_layers=gin.REQUIRED,
            g_num_layers=gin.REQUIRED,
            e_num_filters=gin.REQUIRED,
            g_num_filters=gin.REQUIRED,
            batch_norm=gin.REQUIRED,
            dcgan_mod=gin.REQUIRED,
            filter_size=gin.REQUIRED):

        super(DcganModel, self).__init__()

        self.input_dims = input_dims

        self.e_num_filters = e_num_filters
        self.g_num_filters = g_num_filters
        self.e_num_layers = e_num_layers
        self.g_num_layers = g_num_layers

        self.filter_size = filter_size
        self.dcgan_mod = dcgan_mod

        self.batch_norm = batch_norm
        self.z_dim = z_dim
        self.nc = nc
        self.distribution = distribution
        self.input_normalize_sym = input_normalize_sym

        self.encoder = self.build_encoder_layers()
        self.decoder = self.build_decoder_layers()
        
        self.weight_init()

    def build_encoder_layers(self):
        self.encoder_layers = []
        channels_in = self.input_dims[2]
        map_width = self.input_dims[1]

        for i in range(self.e_num_layers):
            map_width = math.ceil(map_width / 2)
            scale = 2**(self.e_num_layers - i - 1)
            channels_out = int(self.e_num_filters / scale)

            layer = PadSame((self.filter_size, self.filter_size), (2, 2), (1, 1))
            self.encoder_layers.append(layer)

            layer = nn.Conv2d(channels_in, channels_out, self.filter_size, stride=2, bias=True)
            self.encoder_layers.append(layer)

            if self.batch_norm:
                self.encoder_layers.append(nn.BatchNorm2d(channels_out))

            self.encoder_layers.append(nn.ReLU(True))
            channels_in = channels_out

        flatten = nn.Flatten()
        self.encoder_layers.append(flatten)

        z_proj = nn.Linear(int(channels_out * map_width**2), self.z_dim)
        self.encoder_layers.append(z_proj)

        return nn.Sequential(*self.encoder_layers)

    def build_decoder_layers(self):

        self.decoder_layers = []

        if self.dcgan_mod:
            height = math.ceil(self.input_dims[0] / 2**self.g_num_layers)
            width = math.ceil(self.input_dims[1] / 2**self.g_num_layers)
        else:
            height = math.ceil(self.input_dims[0] / 2**(self.g_num_layers - 1)) 
            width = math.ceil(self.input_dims[1] / 2**(self.g_num_layers - 1))

        layer = nn.Linear(self.z_dim, self.g_num_filters * height * width)
        self.decoder_layers.append(layer)

        layer = View((-1, self.g_num_filters, height, width))
        self.decoder_layers.append(layer)

        self.decoder_layers.append(nn.ReLU(True))

        channels_in = self.g_num_filters

        for i in range(self.g_num_layers - 1):
            scale = 2**(i + 1)
            _out_shape = [height * scale, width * scale, int(self.g_num_filters / scale)]
            channels_out = _out_shape[2]

            self.decoder_layers.append(nn.Upsample(scale_factor=2))
            self.decoder_layers.append(PadSame((self.filter_size, self.filter_size), (1, 1), (1, 1)))
            self.decoder_layers.append(nn.Conv2d(channels_in, channels_out, self.filter_size, stride=1))

            if self.batch_norm:
                self.decoder_layers.append(nn.BatchNorm2d(channels_out))

            self.decoder_layers.append(nn.ReLU(True))
            channels_in = channels_out

        self.decoder_layers.append(nn.Upsample(scale_factor=2))
        if self.dcgan_mod:
            self.decoder_layers.append(PadSame((self.filter_size, self.filter_size), (1, 1), (1, 1)))
            layer = nn.Conv2d(channels_in, self.input_dims[-1], self.filter_size, stride=1)
        else:
            self.decoder_layers.append(PadSame((self.filter_size, self.filter_size), (2, 2), (1, 1)))
            layer = nn.Conv2d(channels_in, self.input_dims[-1], self.filter_size, stride=2)
        self.decoder_layers.append(layer)

        return nn.Sequential(*self.decoder_layers)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)

        return x_recon, z

    def _encode(self, x):

        #for layer in self.encoder_layers:
        #    x = layer(x)
        #    print(x.size())
        x = self.encoder(x)

        if self.distribution == "sphere":
            x = F.normalize(x, dim=1, p=2)

        return x

    def _decode(self, z):

        xd = z
        #for layer in self.decoder_layers:
        #   xd = layer(xd)
        #   print(xd.size())

        xd = self.decoder(xd)

        if self.input_normalize_sym:
            return F.tanh(xd)
        else:
            return F.sigmoid(xd)


@gin.configurable
class MlpModel(nn.Module):
    """Encoder-Decoder architecture for MINST-like datasets."""
    def __init__(self, z_dim=10, input_dims=(32, 32, 3), distribution=gin.REQUIRED, input_normalize_sym=gin.REQUIRED, encoder_layer_dims=gin.REQUIRED, decoder_layer_dims=gin.REQUIRED, batch_norm=gin.REQUIRED):
        super(MnistModel, self).__init__()
        self.input_dims = input_dims
        self.encoder_layer_dims = encoder_layer_dims
        self.decoder_layer_dims = decoder_layer_dims
        self.batch_norm = batch_norm
        self.z_dim = z_dim
        self.distribution = distribution
        self.input_normalize_sym = input_normalize_sym

        self.encoder = self.build_encoder_layers()
        self.decoder = self.build_decoder_layers()
        
        self.weight_init()

    def build_encoder(self):
        inp_dims = self.input_dims
        self.encoder_layers = []
        for dims in range(self.encoder_layer_dims):
            layer = nn.Linear(inp_dims, dims)
            self.encoder_layers.append(layer)

            if self.batch_norm:
                self.encoder_layers.append(nn.BatchNorm2d(inp))

            self.encoder_layers.append(nn.ReLU(True))
            inp_dims = dims

        self.encoder_layers.append(nn.Linear(inp, self.z_dim))


    def build_decoder(self):
        inp_dims = self.z_dim
        self.decoder_layers = []
        for dims in range(self.decoder_layer_dims):
            layer = nn.Linear(inp_dims, dims)
            self.decoder_layers.append(layer)

            if self.batch_norm:
                self.decoder_layers.append(nn.BatchNorm2d(inp))

            self.decoder_layers.append(nn.ReLU(True))
            inp_dims = dims

        self.decoder_layers.append(nn.Linear(inp, self.z_dim))


    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)

        return x_recon, z

    def _encode(self, x):

        for layer in self.encoder_layers:
            x = layer(x)

        if self.distribution == "sphere":
            x = F.normalize(x, dim=1, p=2)

        return x

    def _decode(self, z):

        xd = z
        for layer in self.decoder_layers:
            xd = layer(xd)

        if self.input_normalize_sym:
            return F.tanh(xd)
        else:
            return F.sigmoid(xd)


@gin.configurable
class MnistModel(nn.Module):
    """Encoder-Decoder architecture for MINST-like datasets."""
    def __init__(self, z_dim=10, nc=1, input_size=(28,28,1), distribution = gin.REQUIRED, input_normalize_sym=False):
        super(MnistModel, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.distribution = distribution
        self.input_normalize_sym = input_normalize_sym
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1, bias=False),              # B,  128, 14, 14
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),              # B,  256, 7, 7
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            View((-1, 64*7*7)),                                   # B, 64*4*4
            nn.Linear(64*7*7, z_dim)                            # B, z_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64*7*7),                           # B, 64*7*7
            View((-1, 64, 7, 7)),                               # B, 64,  7,  7
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),   # B,  64, 14, 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, 4, 2, 1, bias=False),    # B,  256, 28, 28
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)

        return x_recon, z

    def _encode(self, x):
        #distribution = gin.REQUIRED
        if self.distribution == "sphere":
            return F.normalize(self.encoder(x), dim=1, p=2)
        else:
            return self.encoder(x)

    def _decode(self, z):
        xd = self.decoder(z)
        if self.input_normalize_sym:
            return F.tanh(xd)
        else:
            return F.sigmoid(xd)


class Adversary(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Adversary, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),                                # B, 512
            nn.ReLU(True),
            nn.Linear(512, 512),                                  # B, 512
            nn.ReLU(True),
            nn.Linear(512, 512),                                  # B, 512
            nn.ReLU(True),
            nn.Linear(512, 512),                                  # B, 512
            nn.ReLU(True),
            nn.Linear(512, 1),                                    # B,   1
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    pass
