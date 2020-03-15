import argparse
import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.utils.data as data
from torch import nn, optim

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from data import LRW

parser = argparse.ArgumentParser(description='Lip Generator Example')
parser.add_argument('--data', type=str, default='/beegfs/cy1355/lipread_datachunk_big/', metavar='N',
                    help='data root directory')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=1001, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

def main():
    dataset_list = ['train', 'val', 'test']
    print("loading data...")
    dsets = {x: LRW(x, root_path = args.data) for x in dataset_list}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,\
                        shuffle=True, **kwargs) \
                        for x in dataset_list}
    train_loader = dset_loaders['train']
    val_loader = dset_loaders['val']
    test_loader = dset_loaders['test']
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))

    epochs = 1000
    for epoch in range(0, epochs):
        train(epoch, train_loader)
    


def train(epoch, train_loader):
    discriminator.train()

    train_loss = 0
    train_loader.dataset.test_case = False

    for batch_idx, ((keypoints, mfcc), _) in enumerate(train_loader):

        batch_size = keypoints.shape[0]
        video_length = keypoints.shape[1]

        discriminator_optimizer.zero_grad()

        keypoints = keypoints.to(device)
        mfcc = mfcc.transpose(1,2).to(device).view(batch_size, video_length, 12)

        mouth_points = keypoints[:, : ,48:68].view(batch_size, video_length, 40)

        shuffle_index = torch.randperm(batch_size)
        shuffled_mouth_points = mouth_points[shuffle_index]

        mixed_mouth_points = torch.cat((mouth_points, shuffled_mouth_points), dim = 0).view(-1, 40)
        doubled_mfcc = torch.cat((mfcc, mfcc), dim = 0).view(-1, 12)

        mixed_align_pred = discriminator(mixed_mouth_points, doubled_mfcc) 

        correct_pred = mixed_align_pred[:batch_size * video_length]
        wrong_pred = mixed_align_pred[batch_size * video_length:]
        loss = 1 - correct_pred.mean() + wrong_pred.mean()

        loss.backward()

        train_loss += loss.item()

        discriminator_optimizer.step()

        if batch_idx % args.log_interval == 0:
            
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx * args.batch_size / len(train_loader.dataset),
                loss.item() / len(mfcc)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.mouth_encoder = nn.Sequential(
            nn.Linear(40, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
            )

        self.audio_encoder = nn.Sequential(
            nn.Linear(12, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
            )

        self.linear = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
            )

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, video_input, audio_input):
        x_video = self.mouth_encoder(video_input)
        x_audio = self.audio_encoder(audio_input)
        x = torch.norm(x_video - x_audio, 2, dim = 1, keepdim = True)
        return self.linear(x)

discriminator = Discriminator().to(device)

discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))

if __name__ == "__main__":
    main()
