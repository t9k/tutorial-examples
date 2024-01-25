import argparse
import logging
import os
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from t9k import em

parser = argparse.ArgumentParser(
    description='Recording DDP training of PyTorch model for MNIST with EM.')
parser.add_argument('--ais_host', type=str, help='URL of AIStore server.')
parser.add_argument('--api_key', type=str, help='API Key of user.')
parser.add_argument(
    '--backend',
    type=str,
    help='Distributed backend',
    choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
    default=dist.Backend.GLOO)
parser.add_argument('--log_dir',
                    type=str,
                    help='Path of the TensorBoard log directory.')
parser.add_argument('--save_path',
                    type=str,
                    help='Path of the saved model.')
parser.add_argument('--no_cuda',
                    action='store_true',
                    default=False,
                    help='Disable CUDA training.')
logging.basicConfig(format='%(message)s', level=logging.INFO)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hparams['conv_channels1'],
                               hparams['conv_kernel_size'], 1)
        self.conv2 = nn.Conv2d(hparams['conv_channels1'],
                               hparams['conv_channels2'],
                               hparams['conv_kernel_size'], 1)
        self.conv3 = nn.Conv2d(hparams['conv_channels2'],
                               hparams['conv_channels3'],
                               hparams['conv_kernel_size'], 1)
        self.pool = nn.MaxPool2d(hparams['maxpool_size'],
                                 hparams['maxpool_size'])
        self.dense1 = nn.Linear(576, hparams['linear_features1'])
        self.dense2 = nn.Linear(hparams['linear_features1'], 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        output = F.softmax(self.dense2(x), dim=1)
        return output


def train():
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

            if step % (500 // world_size) == 0:
                train_loss = loss.item()
                logging.info(
                    'epoch {:d}/{:d}, batch {:5d}/{:d} with loss: {:.4f}'.
                    format(epoch, epochs, step, steps_per_epoch, train_loss))
                global_step = (epoch - 1) * steps_per_epoch + step

                if args.log_dir and rank == 0:
                    writer.add_scalar('train/loss', train_loss, global_step)

                if rank == 0:
                    run.log(type='train',
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

    if args.log_dir and rank == 0:
        writer.add_scalar('{:s}/loss'.format(label), test_loss, global_step)
        writer.add_scalar('{:s}/accuracy'.format(label), test_accuracy,
                          global_step)

    if rank == 0:
        run.log(type=label,
                metrics={
                    'loss': test_loss,
                    'accuracy': test_accuracy,
                },
                step=global_step,
                epoch=epoch)


if __name__ == '__main__':
    args = parser.parse_args()

    logging.info('Using distributed PyTorch with %s backend', args.backend)
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        logging.info('Using CUDA')
    device = torch.device('cuda:{}'.format(local_rank) if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    hparams = {
        'batch_size': 32 * world_size,
        'epochs': 10,
        'learning_rate': 0.001 * world_size,
        'learning_rate_decay_period': 1,
        'learning_rate_decay_factor': 0.7,
        'conv_channels1': 32,
        'conv_channels2': 64,
        'conv_channels3': 64,
        'conv_kernel_size': 3,
        'maxpool_size': 2,
        'linear_features1': 64,
        'seed': 1,
    }

    if rank == 0:
        run = em.create_run(name='mnist_torch_distributed')
        run.hparams.update(hparams)

    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'data')
    # rank 0 downloads datasets in advance
    if rank == 0:
        datasets.MNIST(root=dataset_path, train=True, download=True)

    torch.manual_seed(hparams['seed'])

    model = Net().to(device)
    model = DDP(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hparams['learning_rate_decay_period'],
        gamma=hparams['learning_rate_decay_factor'])

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])
    train_dataset = datasets.MNIST(root=dataset_path,
                                   train=True,
                                   download=False,
                                   transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [48000, 12000])
    test_dataset = datasets.MNIST(root=dataset_path,
                                  train=False,
                                  download=False,
                                  transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams['batch_size'],
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

    if args.log_dir and rank == 0:
        if os.path.exists(args.log_dir):
            shutil.rmtree(args.log_dir, ignore_errors=True)
        writer = SummaryWriter(args.log_dir)

    global_step = 0
    epochs = hparams['epochs']
    steps_per_epoch = len(train_loader)
    train()
    test()

    if rank == 0:
        torch.save(model.state_dict(), args.save_path)
        model_artifact = em.create_artifact(name='mnist_torch_saved_model')
        model_artifact.add_file(args.save_path)
        run.mark_output(model_artifact)

        run.finish()
        em.login(ais_host=args.ais_host, api_key=args.api_key)
        run.upload(folder='em-examples', make_folder=True)
