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
    
    input_mouth = torch.rand(29, 40)
    input_audio = torch.rand(29, 12)
    mouth_feature = mouth_encoder(input_mouth)
    audio_feature = audio_encoder(input_audio)
    result = discriminator(mouth_feature, audio_feature)
    import pdb; pdb.set_trace()


def train(epoch, train_loader):
    mouth_encoder.train()
    audio_encoder.train()

    train_loss = 0
    train_loader.dataset.test_case = False

    for batch_idx, ((keypoints, mfcc), _) in enumerate(train_loader):

        batch_size = keypoints.shape[0]
        video_length = keypoints.shape[1]

        encoders_optimizer.zero_grad()

        keypoints = keypoints.to(device)
        mfcc = mfcc.transpose(1,2).to(device).view(-1, 12)

        mouth_points = keypoints[:, : ,48:68].view(-1, 40)

        mouth_embedding = mouth_encoder(mouth_points)
        audio_embedding = audio_encoder(mfcc)

        # Shuffle mouth_embedding
        shuffle_index = torch.randperm(batch_size)
        mouth_embedding_extended = mouth_embedding.view(batch_size, video_length, -1)
        shuffled_mouth_embedding = mouth_embedding_extended[shuffle_index].view(batch_size * video_length, -1)

        mixed_mouth_embedding = torch.cat((mouth_embedding, shuffled_mouth_embedding), dim = 0)
        doubled_audio_embedding = torch.cat((audio_embedding, audio_embedding), dim = 0)

        #mixed_embedding = torch.cat((mixed_mouth_embedding, doubled_audio_embedding), dim = 1)

        mixed_align_pred = discriminator(mixed_mouth_embedding, doubled_audio_embedding) 

        correct_pred = mixed_align_pred[:batch_size * video_length]
        wrong_pred = mixed_align_pred[batch_size * video_length:]
        loss = 1 - correct_pred.mean() + wrong_pred.mean()

        loss.backward()

        train_loss += loss.item()

        encoders_optimizer.step()

        if batch_idx % args.log_interval == 0:
            
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx * args.batch_size / len(train_loader.dataset),
                loss.item() / len(mfcc)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MouthEncoder(nn.Module):
    def __init__(self):
        super(MouthEncoder, self).__init__()

        self.encoder = nn.Sequential(
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

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.encoder(x)

mouth_encoder = MouthEncoder().to(device)

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.encoder = nn.Sequential(
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

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.encoder(x)

audio_encoder = AudioEncoder().to(device)

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

    def forward(self, x_video, x_audio):
        x = torch.norm(x_video - x_audio, 2, dim = 1, keepdim = True)
        return self.linear(x)

discriminator = Discriminator().to(device)

encoders_params = list(mouth_encoder.parameters()) + list(audio_encoder.parameters()) + list(discriminator.parameters())
# TO DO: Check for beta parameters for adam optimizer.
encoders_optimizer = optim.Adam(encoders_params, lr=1e-3, betas=(0.5, 0.999))

if __name__ == "__main__":
    main()
