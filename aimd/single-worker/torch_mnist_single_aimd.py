import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from t9k import aimd

parser = argparse.ArgumentParser(
    description=
    'Recording of training data of PyTorch model for MNIST with AIMD.')
parser.add_argument('--aimd_host',
                    type=str,
                    required=True,
                    help='URL of AIMD server.')
parser.add_argument('--api_key',
                    type=str,
                    required=True,
                    help='API Key for communicating with AIMD server.')
parser.add_argument('--no_cuda',
                    action='store_true',
                    default=False,
                    help='Disable CUDA training.')
logging.basicConfig(format='%(message)s', level=logging.INFO)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, params['conv_channels1'],
                               params['conv_kernel_size'], 1)
        self.conv2 = nn.Conv2d(params['conv_channels1'],
                               params['conv_channels2'],
                               params['conv_kernel_size'], 1)
        self.conv3 = nn.Conv2d(params['conv_channels2'],
                               params['conv_channels3'],
                               params['conv_kernel_size'], 1)
        self.pool = nn.MaxPool2d(params['maxpool_size'],
                                 params['maxpool_size'])
        self.dense1 = nn.Linear(576, params['linear_features1'])
        self.dense2 = nn.Linear(params['linear_features1'], 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        output = F.softmax(self.dense2(x), dim=1)
        return output


def train(scheduler):
    global global_step
    for epoch in range(1, epochs + 1):
        model.train()
        for step, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                train_loss = loss.item()
                logging.info(
                    'epoch {:d}/{:d}, batch {:5d}/{:d} with loss: {:.4f}'.
                    format(epoch, epochs, step, steps_per_epoch, train_loss))
                global_step = (epoch - 1) * steps_per_epoch + step

                trial.log(
                    metrics_type='train',
                    metrics={'loss': train_loss},
                    step=global_step,
                    epoch=epoch)

        scheduler.step()
        global_step = epoch * steps_per_epoch
        test(val=True, epoch=epoch)


def test(val=False, epoch=None):
    label = 'val' if val else 'test'
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        loader = val_loader if val else test_loader
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            prediction = output.max(1)[1]
            correct += (prediction == target).sum().item()

    test_loss = running_loss / len(loader)
    test_accuracy = correct / len(loader.dataset)
    msg = '{:s} loss: {:.4f}, {:s} accuracy: {:.4f}'.format(
        label, test_loss, label, test_accuracy)
    if val:
        msg = 'epoch {:d}/{:d} with '.format(epoch, epochs) + msg
    logging.info(msg)

    trial.log(
        metrics_type=label,
        metrics={
            'loss': test_loss,
            'accuracy': test_accuracy,
        },
        step=global_step,
        epoch=epoch)


if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        logging.info('Using CUDA')
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    trial = aimd.create_trial(
        trial_name='mnist_torch',
        folder_path='aimd-example',
    )

    trial.params.update({
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.001,
        'learning_rate_decay_period': 1,
        'learning_rate_decay_factor': 0.7,
        'conv_channels1': 32,
        'conv_channels2': 64,
        'conv_channels3': 64,
        'conv_kernel_size': 3,
        'maxpool_size': 2,
        'linear_features1': 64,
        'seed': 1,
    })
    params = trial.params

    torch.manual_seed(params['seed'])

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params['learning_rate_decay_period'],
        gamma=params['learning_rate_decay_factor'])

    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'data')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])
    train_dataset = datasets.MNIST(root=dataset_path,
                                   train=True,
                                   download=True,
                                   transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [48000, 12000])
    test_dataset = datasets.MNIST(root=dataset_path,
                                  train=False,
                                  download=True,
                                  transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=params['batch_size'],
                                               shuffle=True,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=400,
                                             shuffle=False,
                                             **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1000,
                                              shuffle=False,
                                              **kwargs)

    global_step = 0
    epochs = params['epochs']
    steps_per_epoch = len(train_loader)
    train(scheduler)
    test()

    trial.finish()
    aimd.login(host=args.aimd_host, api_key=args.api_key)
    trial.upload(make_folder=True)
