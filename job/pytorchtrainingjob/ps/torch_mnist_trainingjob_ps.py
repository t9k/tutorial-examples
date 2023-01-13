import argparse
import logging
import os
import shutil
from threading import Lock
import time

import torch
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

param_server = None
global_lock = Lock()

parser = argparse.ArgumentParser(
    description='Distributed training of PyTorch model for MNIST '
    'with RPC-based parameter server.')
parser.add_argument('--log_dir',
                    type=str,
                    help='Path of the TensorBoard log directory.')
parser.add_argument('--no_cuda',
                    action='store_true',
                    default=False,
                    help='Disable CUDA training.')
parser.add_argument('--save_path',
                    type=str,
                    default=None,
                    help='Save path of the trained model.')
logger = logging.getLogger('print')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False


# --------- MNIST Network to train, from pytorch/examples -----
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


# --------- Parameter Server --------------------
class ParameterServer(nn.Module):

    def __init__(self):
        super().__init__()
        self.models = {}
        # This lock is only used during init, and does not
        # impact training perf.
        self.models_init_lock = Lock()
        self.use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.num_cuda = torch.cuda.device_count() if self.use_cuda else 0
        logger.info('Using %s GPUs for training', self.num_cuda)

    def forward(self, rank, inp):
        inp = inp.to(self.get_device(rank))
        out = self.models[rank](inp)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        out = out.to('cpu')
        return out

    # Use dist autograd to retrieve gradients accumulated for this model.
    # Primarily used for verification.
    def get_dist_gradients(self, cid):
        grads = dist_autograd.get_gradients(cid)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k.to('cpu'), v.to('cpu')
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes parameters remotely.
    def get_param_rrefs(self, rank):
        param_rrefs = [
            rpc.RRef(param) for param in self.models[rank].parameters()
        ]
        return param_rrefs

    def create_model_for_rank(self, rank):
        with self.models_init_lock:
            if rank not in self.models:
                torch.manual_seed(params['seed'])
                device = self.get_device(rank)
                self.models[rank] = Net().to(device)
                logger.info('Putting model of worker %s on device %s', rank,
                            device)

    def get_num_models(self):
        with self.models_init_lock:
            return len(self.models)

    def average_model(self, rank):
        # Load state dict of requested rank
        state_dict_for_rank = self.models[rank].state_dict()
        device = self.get_device(rank)
        # Average all params
        for key in state_dict_for_rank:
            state_dict_for_rank[key] = sum(
                self.models[r].state_dict()[key].to(device)
                for r in self.models) / len(self.models)
        # Rewrite back state dict
        self.models[rank].load_state_dict(state_dict_for_rank)

    def save_model(self, rank, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(self.models[rank].state_dict(), path)

    def train(self, rank):
        self.models[rank].train()

    def eval(self, rank):
        self.models[rank].eval()

    def get_device(self, rank):
        if not self.num_cuda:
            device = 'cpu'
        elif rank <= self.num_cuda:
            device = rank - 1
        else:
            device = self.num_cuda - 1
        return device


def get_parameter_server(rank):
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer()
        # Add model for this rank
        param_server.create_model_for_rank(rank)
        return param_server


def run_parameter_server():
    # The parameter server just acts as a host for the model and responds to
    # requests from workers, hence it does not need to run a loop.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all workers
    # to complete, and then exit.
    logger.info('PS master initializing RPC')
    rpc.init_rpc(name='parameter_server', rank=rank, world_size=world_size)
    logger.info('RPC initialized! Running parameter server...')
    rpc.shutdown()
    logger.info('RPC shutdown on parameter server.')


# --------- workers --------------------
# nn.Module corresponding to the network trained by this worker. The
# forward() method simply invokes the network on the given parameter
# server.
class TrainerNet(nn.Module):

    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.param_server_rref = rpc.remote('parameter_server',
                                            get_parameter_server,
                                            args=(self.rank, ))

    def get_global_param_rrefs(self):
        remote_params = self.param_server_rref.rpc_sync().get_param_rrefs(
            self.rank)
        return remote_params

    def forward(self, x):
        model_output = self.param_server_rref.rpc_sync().forward(self.rank, x)
        return model_output

    def average_model(self):
        self.param_server_rref.rpc_sync().average_model(self.rank)

    def save_model(self, path):
        self.param_server_rref.rpc_sync().save_model(self.rank, path)

    def train(self):
        self.param_server_rref.rpc_sync().train(self.rank)

    def eval(self):
        self.param_server_rref.rpc_sync().eval(self.rank)


def train(net):
    # Wait for all nets on PS to be created, otherwise we could run
    # into race conditions during training.
    num_created = net.param_server_rref.rpc_sync().get_num_models()
    while num_created != world_size - 1:
        time.sleep(0.5)
        num_created = net.param_server_rref.rpc_sync().get_num_models()

    # Build DistributedOptimizer.
    param_rrefs = net.get_global_param_rrefs()
    optimizer = DistributedOptimizer(optim.Adam,
                                     param_rrefs,
                                     lr=params['learning_rate'])
    # Wait for official support for lr_scheduler for distributed optimizer

    global global_step

    # Runs the typical neural network forward + backward + optimizer step, but
    # in a distributed fashion.
    for epoch in range(1, epochs + 1):
        net.train()
        for step, (data, target) in enumerate(train_loader, 1):
            with dist_autograd.context() as cid:
                output = net(data)
                loss = F.cross_entropy(output, target)
                dist_autograd.backward(cid, [loss])
                # Ensure that dist autograd ran successfully and gradients were
                # returned.
                assert net.param_server_rref.rpc_sync().get_dist_gradients(
                    cid) != {}
                optimizer.step(cid)

            if step % 25 == 0:
                # Request server to update model with average params across all
                # workers.
                net.average_model()

            if step % 500 == 0:
                train_loss = loss.item()
                logger.info('epoch %d/%d, batch %5d/%d with loss: %.4f', epoch,
                            epochs, step, steps_per_epoch, train_loss)
                global_step = (epoch - 1) * steps_per_epoch + step

                if args.log_dir:
                    writer.add_scalar('train/loss', train_loss, global_step)

        global_step = epoch * steps_per_epoch
        test(net, val=True, epoch=epoch)


def test(net, val=False, epoch=None):
    label = 'val' if val else 'test'
    net.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        loader = val_loader if val else test_loader
        for data, target in loader:
            output = net(data)
            loss = F.cross_entropy(output, target)
            running_loss += loss.item()
            prediction = output.max(1)[1]
            correct += (prediction == target).sum().item()

    test_loss = running_loss / len(loader)
    test_accuracy = correct / len(loader.dataset)
    msg = '{:s} loss: {:.4f}, {:s} accuracy: {:.4f}'.format(
        label, test_loss, label, test_accuracy)
    if val:
        msg = 'epoch {:d}/{:d} with '.format(epoch, epochs) + msg
    logger.info(msg)

    if args.log_dir:
        writer.add_scalar('{:s}/loss'.format(label), test_loss, global_step)
        writer.add_scalar('{:s}/accuracy'.format(label), test_accuracy,
                          global_step)


# Main loop for workers.
def run_worker():
    logger.info('Worker with rank %d initializing RPC', rank)
    rpc.init_rpc(name='worker_{}'.format(rank),
                 rank=rank,
                 world_size=world_size)
    logger.info('Worker with rank %d done initializing RPC', rank)

    global train_loader, val_loader, test_loader, steps_per_epoch
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])
    train_set = datasets.MNIST(root=dataset_path,
                               train=True,
                               download=False,
                               transform=transform)
    train_set, val_set = torch.utils.data.random_split(train_set,
                                                       [48000, 12000])
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=params['batch_size'],
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=500,
                                             shuffle=False)
    test_set = datasets.MNIST(root=dataset_path,
                              train=False,
                              download=False,
                              transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=500,
                                              shuffle=False)
    steps_per_epoch = len(train_loader)

    net = TrainerNet(rank=rank)
    train(net)
    test(net)

    if args.save_path:
        path = args.save_path + '.{}'.format(rank)
        net.save_model(path)

    rpc.shutdown()


# --------- Launcher --------------------
if __name__ == '__main__':
    args = parser.parse_args()

    rank = int(os.getenv('RANK'))
    world_size = int(os.getenv('WORLD_SIZE'))

    params = {
        'batch_size': 32,
        'epochs': 5,  # due to no scheduler
        'learning_rate': 0.001 * (world_size - 1),
        'conv_channels1': 32,
        'conv_channels2': 64,
        'conv_channels3': 64,
        'conv_kernel_size': 3,
        'maxpool_size': 2,
        'linear_features1': 64,
        'seed': 1,
    }

    torch.manual_seed(params['seed'])

    if rank == 0:
        run_parameter_server()
    else:
        dataset_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data')
        if rank == 1:
            datasets.MNIST(root=dataset_path, train=True, download=True)
            datasets.MNIST(root=dataset_path, train=False, download=True)
        train_loader = None
        val_loader = None
        test_loader = None

        if args.log_dir:
            log_dir = os.path.join(args.log_dir, 'worker_' + str(rank))
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir, ignore_errors=True)
            writer = SummaryWriter(log_dir)

        global_step = 0
        epochs = params['epochs']
        steps_per_epoch = None
        run_worker()
