__author__ = 'voanna'

import os
import numpy as np
import io
import glob
from PIL import Image
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils import tensorboard
import torchvision
import functools
import matplotlib
import collections

matplotlib.use('Agg')


np.seterr(all='raise')
np.random.seed(2019)
torch.manual_seed(2019)

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save_epoch_interval', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--latent-dimension', type=int, default=256, metavar='N',
                    help=' ')
parser.add_argument('--n-channels', type=int, default=1, metavar='N',
                    help=' ')
parser.add_argument('--img-size', type=int, default=256, metavar='N',
                    help=' ')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='N',
                    help=' ')
parser.add_argument('--architecture', type=str, default='old', metavar='N',
                    help=' ')
parser.add_argument('--reconstruction-data-loss-weight', type=float, default=1.0,
                    help=' ')
parser.add_argument('--kl-latent-loss-weight', type=float, default=0.01,
                    help=' ')
parser.add_argument("--mri_data_dir", type = str, 
                    default ="your_data_path",
                    help = "the data directory")

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


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


# Reconstruction + KL divergence losses summed over all elements and batch
reconstruction_loss = nn.MSELoss(reduction='sum')


def KLLoss(z_mean, z_log_sigma):
    return -0.5 * torch.sum(
        1 + 2 * z_log_sigma - torch.pow(z_mean, 2) -
        torch.exp(2 * z_log_sigma))




def add_grid(writer, images, name, step,
             batch_size=32, n_channels=1, img_size=128):
    grid = torchvision.utils.make_grid(
        images.view(batch_size, n_channels, img_size, img_size),
        normalize=True,
        range=(-1, 1))

    writer.add_image(name, grid, step)


def add_detailed_summaries(writer, decoder, phase, data, reconstruction, latent,
                           step,
                           batch_size=32,
                           n_channels=1,
                           img_size=128):
    add_grid(writer, data, 'Data/{}'.format(phase), step,
             batch_size=batch_size, n_channels=n_channels, img_size=img_size)
    add_grid(writer, reconstruction, 'Reconstruction/{}'.format(phase), step,
             batch_size=batch_size, n_channels=n_channels, img_size=img_size)

    zs = torch.randn_like(latent)
    samples = decoder(zs)
    add_grid(writer, samples, 'Samples/{}'.format(phase), step,
             batch_size=batch_size, n_channels=n_channels, img_size=img_size)


def train(encoder, decoder, optimizer, epoch, step, train_loader, writer,
          reconstruction_data_loss_weight=1.0,
          kl_latent_loss_weight=1.0,
          batch_size=32,
          log_interval=1000,
          n_channels=1,
          img_size=128):
    encoder.train()
    decoder.train()
    train_loss = 0
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device())
        # print(data.shape)
        optimizer.zero_grad()
        latent, mu, std = encoder(data)
        # print(latent.shape)
        reconstruction = decoder(latent)
        # print(reconstruction.shape)
        reconstruction_data_loss = reconstruction_loss(reconstruction, data)
        kl_latent_loss = KLLoss(mu, std)

        loss = reconstruction_data_loss * reconstruction_data_loss_weight
        loss += kl_latent_loss_weight * kl_latent_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        step += 1
        writer.add_scalar('Loss/train_sum', loss.item(), step)
        writer.add_scalar('Loss/train_reconstruction_data_loss', reconstruction_data_loss.item(), step)
        writer.add_scalar('Loss/train_kl_latent_loss', kl_latent_loss.item(), step)

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                       100. * i / len(train_loader),
                       loss.item() / len(data)))
    add_detailed_summaries(writer, decoder, 'train', data, reconstruction, latent,
                           step,
                           batch_size=batch_size,
                           n_channels=n_channels,
                           img_size=img_size)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    return encoder, decoder, optimizer, step


def test(encoder, decoder, epoch, step, test_loader, writer,
         reconstruction_data_loss_weight=1.0,
         kl_latent_loss_weight=1.0,
         batch_size=32,
         n_channels=1,
         img_size=128):
    n_test = 40
    encoder.eval()
    decoder.eval()
    test_loss = 0
    test_reconstruction_data_loss = 0
    test_kl_latent_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device())
            latent, mu, std = encoder(data)
            latent = latent.to(device())
            reconstruction = decoder(latent)
            reconstruction_data_loss = reconstruction_loss(reconstruction, data)
            kl_latent_loss = KLLoss(mu, std)
            loss = reconstruction_data_loss * reconstruction_data_loss_weight
            loss += kl_latent_loss_weight * kl_latent_loss

            test_loss += loss.item()
            test_reconstruction_data_loss += reconstruction_data_loss.item()
            test_kl_latent_loss += kl_latent_loss.item()

    phase = 'test'
    writer.add_scalar('Loss/{}_sum'.format(phase), test_loss / n_test, step)
    writer.add_scalar('Loss/{}_reconstruction_data_loss'.format(phase), test_reconstruction_data_loss / n_test, step)
    writer.add_scalar('Loss/{}_kl_latent_loss'.format(phase),
                      test_kl_latent_loss / n_test, step)

    add_detailed_summaries(writer, decoder, 'test', data, reconstruction, latent,
                           step,
                           batch_size=batch_size,
                           n_channels=n_channels,
                           img_size=img_size)

    writer.flush()
    print('====> Test set loss: {:.4f}'.format(test_loss / n_test))
    return encoder, decoder, step


def make_train_loader(args, batch_size):
    def subsample_volumes(fname):
        basename = os.path.basename(fname)
        id = basename.split('_')[-1]
        id = int(id.split('.')[0])
        return id < batch_size  # id in list(8*np.arange(33))

    train_data = datasets.ImageFolder(args.mri_data_dir + '/train/',
                                      transform=transforms.Compose([transforms.Grayscale(),
                                                                    transforms.ToTensor()]),
                                      # transforms.Normalize([0.5], [0.5])]),
                                      is_valid_file=subsample_volumes)
    return torch.utils.data.DataLoader(train_data,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=4,
                                       drop_last=True)

def make_test_loader(args, batch_size):
    def subsample_volumes(fname):
        basename = os.path.basename(fname)
        id = basename.split('_')[-1]
        id = int(id.split('.')[0])
        return id < batch_size

    test_data = datasets.ImageFolder(args.mri_data_dir + '/test/',
                                     transform=transforms.Compose([transforms.Grayscale(),
                                                                   transforms.ToTensor()]),
                                     # transforms.Normalize([0.5], [0.5])]),
                                     is_valid_file=subsample_volumes)

    return torch.utils.data.DataLoader(test_data,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=4,
                                       drop_last=True)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(args):
    torch.manual_seed(args.seed)
    encoder = Encoder(args.n_channels, gf_dim=4)
    decoder = Decoder(args.n_channels, gf_dim=4)

    encoder = encoder.to(device())
    decoder = decoder.to(device())

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)
    print('Cuda is {}available'.format('' if torch.cuda.is_available() else 'not '))

    EXPERIMENT = 'MICCAI-release-version'
    run_name = 'vol_256_lr_{}' \
               '_kl_{}_' \
               '_bsize_{}' \
               ''.format(
        args.learning_rate,
        args.kl_latent_loss_weight,
        args.batch_size)

    load_from_ckpt_dir = os.path.join('experiments', EXPERIMENT, 'gen', run_name, 'checkpoints')
    os.makedirs(load_from_ckpt_dir, exist_ok=True)

    print('testing complete')

    if os.listdir(load_from_ckpt_dir):
        resume_path = os.path.join(load_from_ckpt_dir, sorted(os.listdir(load_from_ckpt_dir))[-1])
        if not os.path.exists(resume_path):
            print("=> no checkpoint found")
            start_epoch = 1
            step = 0
        else:
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=device())
            start_epoch = checkpoint['epoch']
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            step = checkpoint['step']
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found")
        start_epoch = 1
        step = 0

    ckpt_dir = os.path.join('experiments', EXPERIMENT, 'gen', run_name, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    tb_dir = os.path.join('experiments', EXPERIMENT, 'gen', 'tensorboard', run_name)
    if not os.path.isdir(tb_dir):
        os.makedirs(tb_dir, exist_ok=True)

    frame_dir = os.path.join('experiments', EXPERIMENT, 'gen', run_name, 'frames')
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)

    results_dir = os.path.join('experiments', EXPERIMENT, 'gen', run_name, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    print(encoder)
    print(decoder)

    writer = tensorboard.SummaryWriter(log_dir=tb_dir)
    train_loader = make_train_loader(args, batch_size=args.batch_size)
    # test_loader = make_test_loader(batch_size=args.batch_size)
    # encoder, decoder, test_loss = test(encoder, decoder, start_epoch - 1, step, test_loader, writer,
    #                                    reconstruction_data_loss_weight=args.reconstruction_data_loss_weight,
    #                                    kl_latent_loss_weight=args.kl_latent_loss_weight,
    #                                    batch_size=args.batch_size,
    #                                    n_channels=args.n_channels,
    #                                    img_size=args.img_size)

    for epoch in range(start_epoch, args.epochs + 1):
        encoder, decoder, optimizer, step = train(encoder, decoder, optimizer, epoch, step, train_loader,
                                                  writer,
                                                  reconstruction_data_loss_weight=args.reconstruction_data_loss_weight,
                                                  kl_latent_loss_weight=args.kl_latent_loss_weight,
                                                  batch_size=args.batch_size,
                                                  log_interval=args.log_interval,
                                                  n_channels=args.n_channels,
                                                  img_size=args.img_size)

        # encoder, decoder, test_loss = test(encoder, decoder, epoch, step, test_loader, writer,
        #                                    reconstruction_data_loss_weight=args.reconstruction_data_loss_weight,
        #                                    kl_latent_loss_weight=args.kl_latent_loss_weight,
        #                                    batch_size=args.batch_size,
        #                                    n_channels=args.n_channels,
        #                                    img_size=args.img_size)

        if epoch % args.save_epoch_interval == 0:
            save_path = os.path.join(ckpt_dir, 'model_{0:08d}.pth.tar'.format(epoch))
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'step': step
            },
                save_path)
            print('Saved model')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)