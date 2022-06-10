import torch
from torch import nn

class ResBlockUp(nn.Module):
    def __init__(self, filters_in, filters_out, act=True):
        super(ResBlockUp, self).__init__()
        self.act = act
        self.conv1_block = nn.Sequential(
            nn.Conv2d(filters_in, filters_in, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(filters_in),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2_block = nn.Sequential(
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(filters_out))

        self.conv3_block = nn.Sequential(
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(filters_out),
            nn.LeakyReLU(0.2, inplace=True))

        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        if self.act:
            conv2 = self.lrelu(conv2)
        conv3 = self.conv3_block(x)
        if self.act:
            conv3 = self.lrelu(conv3)

        return conv2 + conv3


class ResBlockDown(nn.Module):
    def __init__(self, filters_in, filters_out, act=True):
        super(ResBlockDown, self).__init__()
        self.act = act
        self.conv1_block = nn.Sequential(
            nn.Conv2d(filters_in, filters_in, 3, stride=2, padding=1),
            nn.BatchNorm2d(filters_in),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2_block = nn.Sequential(
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(filters_out))

        self.conv3_block = nn.Sequential(
            nn.Conv2d(filters_in, filters_out, 3, stride=2, padding=1),
            nn.BatchNorm2d(filters_out)
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        if self.act:
            conv2 = self.lrelu(conv2)
        conv3 = self.conv3_block(x)
        if self.act:
            conv3 = self.lrelu(conv3)

        return conv2 + conv3


class Encoder(nn.Module):
    def __init__(self, n_channels, gf_dim=16):
        super(Encoder, self).__init__()

        gf_dim = gf_dim

        self.conv1_block = nn.Sequential(
            nn.Conv2d(n_channels, gf_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(gf_dim),
            nn.LeakyReLU(0.2, inplace=True))

        self.res1 = ResBlockDown(gf_dim * 1, gf_dim * 1)
        self.res2 = ResBlockDown(gf_dim * 1, gf_dim * 2)
        self.res3 = ResBlockDown(gf_dim * 2, gf_dim * 4)
        self.res4 = ResBlockDown(gf_dim * 4, gf_dim * 8)
        self.res5 = ResBlockDown(gf_dim * 8, gf_dim * 16)

        self.res6 = ResBlockDown(gf_dim * 16, gf_dim * 32)
        self.res7 = ResBlockDown(gf_dim * 32, gf_dim * 64, act=False)
        self.res2_stdev = ResBlockDown(gf_dim * 32, gf_dim * 64, act=False)

    def encode(self, x):
        
        conv1 = self.conv1_block(x)
        z = self.res1(conv1)
        z = self.res2(z)
        z = self.res3(z)
        z = self.res4(z)
        z = self.res5(z)
        z = self.res6(z)
        z_mean = self.res7(z)

        z_std = self.res2_stdev(z)

        return z_mean, z_std

    def reparameterize(self, mu, std):
        std = torch.exp(std)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)
        return z, mu, std


class Decoder(nn.Module):
    def __init__(self, n_channels, gf_dim=16):
        super(Decoder, self).__init__()


        self.res0 = ResBlockUp(gf_dim*64, gf_dim * 32)
        self.res1 = ResBlockUp(gf_dim*32, gf_dim * 16)
        self.res2 = ResBlockUp(gf_dim*16, gf_dim * 8)
        self.res3 = ResBlockUp(gf_dim*8, gf_dim * 4)
        self.res4 = ResBlockUp(gf_dim*4, gf_dim * 2)
        self.res5 = ResBlockUp(gf_dim*2, gf_dim * 1)
        self.res6 = ResBlockUp(gf_dim*1, gf_dim * 1)
        self.conv_1_block = nn.Sequential(
            nn.Conv2d(gf_dim, gf_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(gf_dim),
            nn.LeakyReLU(0.2, inplace=True))


        self.conv2 = nn.Conv2d(gf_dim, n_channels, 3, stride=1, padding=1)


    def forward(self, z):
        x = self.res0(z)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.conv_1_block(x)
        x = self.conv2(x)
        return x

