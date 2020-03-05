import os
import cv2 
import torch
import argparse
import glob

import numpy as np
import torch.nn.functional as F

from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from data import LRW

from PIL import Image

import datetime

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

print('Run on {}'.format(device))

class FaceEncoder(nn.Module):
    def __init__(self):
        super(FaceEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(96, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            )

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.encoder(x)

face_encoder = FaceEncoder().to(device)

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
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
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
encoders_params = list(face_encoder.parameters()) + list(audio_encoder.parameters())
# TO DO: Check for beta parameters for adam optimizer.
encoders_optimizer = optim.Adam(encoders_params, lr=1e-3, betas=(0.5, 0.999))

class FaceDecoder(nn.Module):
    def __init__(self):
        super(FaceDecoder, self).__init__()
        h_GRU = 144 
        self.stabilizer = nn.GRU(144, h_GRU, 2, batch_first = True, dropout = 0.2)

        self.decoder = nn.Sequential(
            nn.Linear(144, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 40),
            nn.Sigmoid(),
            )

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x, _ = self.stabilizer(x)
        return self.decoder(x.reshape(-1, 144))

face_decoder = FaceDecoder().to(device)
decoder_optimizer = optim.Adam(face_decoder.parameters(), lr=1e-3, betas=(0.5, 0.999))

mse_loss = torch.nn.MSELoss()


dsets = {x: LRW(x, root_path = args.data) for x in ['train', 'val', 'test']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,\
                       shuffle=True, **kwargs) \
                       for x in ['train', 'val', 'test']}
train_loader = dset_loaders['train']
val_loader = dset_loaders['val']
test_loader = dset_loaders['test']
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))

def open_level(mouth_keypoints):
    u_index = [1, 2, 3, 4, 5, 13, 14, 15]
    l_index = [11, 10, 9, 8, 7, 19, 18, 17]
    upper_lip = mouth_keypoints[:, :, u_index, :]
    lower_lip = mouth_keypoints[:, :, l_index, :]

    # Coordinates
    u_x = upper_lip[:, :, :, 0]
    u_y = upper_lip[:, :, :, 1]
    l_x = lower_lip[:, :, :, 0]
    l_y = lower_lip[:, :, :, 1]

    distance = ((u_x - l_x) ** 2 + (u_y - l_y) ** 2) ** 0.5
    distance_mean = distance.mean(dim = 2)
    distance_normed = distance_mean / distance_mean.sum(dim = 1).unsqueeze(1)
    return distance_normed * 256

def train(epoch):
    face_encoder.train()
    audio_encoder.train()
    face_decoder.train()

    train_loss = 0
    train_loader.dataset.test_case = False

    for batch_idx, ((keypoints, mfcc), _) in enumerate(train_loader):

        batch_size = keypoints.shape[0]
        video_length = keypoints.shape[1]

        encoders_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        keypoints = keypoints.to(device)
        mfcc = mfcc.transpose(1,2).to(device).view(-1, 12)

        face_points = keypoints[:, :, :48].view(-1, 96)
        mouth_points = keypoints[:, : ,48:68].view(-1, 40)

        face_embedding = face_encoder(face_points)
        audio_embedding = audio_encoder(mfcc)

        # Shuffle face_embedding
        shuffle_index = torch.randperm(batch_size)
        face_embedding_extended = face_embedding.view(batch_size, video_length, -1)
        shuffled_face_embedding = face_embedding_extended[shuffle_index].view(batch_size * video_length, -1)

        mixed_face_embedding = torch.cat((face_embedding, shuffled_face_embedding), dim = 0)
        doubled_audio_embedding = torch.cat((audio_embedding, audio_embedding), dim = 0)

        mixed_embedding = torch.cat((mixed_face_embedding, doubled_audio_embedding), dim = 1).view(batch_size * 2, -1, 144)

        mixed_mouth_points_pred = face_decoder(mixed_embedding) * 255

        mouth_points_pred = mixed_mouth_points_pred[:batch_size * video_length]
        supervised_loss = mse_loss(mouth_points, mouth_points_pred)

        shuffled_pred = mixed_mouth_points_pred[batch_size * video_length:].view(batch_size, video_length, 20, 2)
        open_score_shuffled = open_level(shuffled_pred)
        original_pred = mouth_points_pred.view(batch_size, video_length, 20, 2)
        open_score_normal = open_level(original_pred)

        #kld = torch.nn.KLDivLoss(reduction = 'batchmean')

        #log_prob = torch.nn.LogSoftmax(dim=1)(open_score_shuffled.transpose(0,1))
        #prob = torch.nn.Softmax(dim=1)(open_score_normal.transpose(0,1))

        adversarial_loss = mse_loss(open_score_shuffled, open_score_normal)
        #adversarial_loss = kld(log_prob, prob)

        loss = supervised_loss + adversarial_loss
        loss.backward()

        train_loss += loss.item()

        encoders_optimizer.step()
        decoder_optimizer.step()

        if batch_idx % args.log_interval == 0:
            
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx * args.batch_size / len(train_loader.dataset),
                loss.item() / len(mfcc)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def save_image_(tensor, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp - A filename(string) or file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def image_from_tensor(tensor):
    batch_size = tensor.shape[0]
    img = np.zeros((batch_size,256,256,3), np.uint8) + 128
    for i in range(batch_size):
        for row in range(67):
            x_1 = tensor[i, row, 0]
            y_1 = tensor[i, row, 1]
            x_2 = tensor[i, row+1, 0]
            y_2 = tensor[i, row+1, 1]
            cv2.circle(img[i], (x_1, y_1), 1, (0, 0, 255), -1)
            cv2.circle(img[i], (x_2, y_2), 1, (0, 0, 255), -1)
            cv2.line(img[i], (x_1, y_1), (x_2, y_2), (0, 0, 255), 1)
    img = np.transpose(img, (0, 3, 1, 2))
    return torch.tensor(img)

def test(epoch):

    face_encoder.eval()
    audio_encoder.eval()
    face_decoder.eval()

    test_loss = 0

    test_loader.dataset.test_case = True

    with torch.no_grad():
        for batch_idx, ((keypoints, mfcc), _) in enumerate(test_loader):

            batch_size = keypoints.shape[0]
            
            keypoints = keypoints.to(device)
            mfcc = mfcc.transpose(1,2).to(device).reshape(-1, 12)

            face_points = keypoints[:, :, :48].reshape(-1, 96)
            mouth_points = keypoints[:, : ,48:68].reshape(-1, 40)

            face_embedding = face_encoder(face_points)
            audio_embedding = audio_encoder(mfcc)

            embedding = torch.cat((face_embedding, audio_embedding), dim = 1).view(batch_size, -1, 144)

            mouth_points_pred = face_decoder(embedding) * 255

            test_loss += mse_loss(mouth_points, mouth_points_pred)

            # if epoch % 10 == 0:
    n = min(keypoints.size(0), 8)
    image_data = image_from_tensor(keypoints[:,0,:,:])
    face_pred = torch.cat((face_points.view(batch_size, 29, 48, 2),
                           mouth_points_pred.view(batch_size, 29, 20, 2)), dim = 2)
    face_pred_batch = image_from_tensor(face_pred[:,0,:,:])
    comparison = torch.cat([image_data[:n], face_pred_batch[:n]], dim = 0)

    if not os.path.exists('./wav_results_base'): os.mkdir('./wav_results_base')
    save_image(comparison.cpu(), './wav_results_base/reconstruction_' + str(epoch) + '_' + str(round(test_loss.item(),6)) + '.png', nrow=n)

    if not os.path.exists('./wav_saves_base'): os.mkdir('./wav_saves_base')
    torch.save(face_encoder, './wav_saves_base/face_encoder_{}.pt'.format(epoch))
    torch.save(audio_encoder, './wav_saves_base/audio_encoder_{}.pt'.format(epoch))
    torch.save(face_decoder, './wav_saves_base/face_decoder_{}.pt'.format(epoch))

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(0, args.epochs):
        print('Epoch {} starts at {}'.format(epoch, datetime.datetime.now()))
        train(epoch)
        test(epoch)
        print('Epoch {} ends at {}'.format(epoch, datetime.datetime.now()))


