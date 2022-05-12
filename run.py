import os
import math
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DistributedSampler
from torch.utils.data.sampler import BatchSampler
from torch.nn.parallel import DistributedDataParallel

import torch.optim as optim
from tqdm import tqdm

EPOCH = 20
LR = 0.001
MMT = 0.9
BS = 16
BS_test = 64

log_internal = 100

'''Define simpmle network
'''

class ClsNet(nn.Module):
    def __init__(self):
        super(ClsNet, self).__init__()
        self.build_net()
        self.init_weights()

    def build_net(self):
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x  = self.conv2(x)
        
        x  = self.conv2_drop(x)
        #    x = x * (1-self.conv2_drop.p)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

mnist_dataset_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,)
                                            )]))
mnist_dataset_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,)
                                            )]))
'''Construct dataset & dataloader
'''
def main_func():
    print('start training....', dist.get_rank())
    #print(len(mnist_dataset)) # --> 60000 
    #print(mnist_dataset[0][0].shape) # 1x28x28, cls

    sampler = DistributedSampler(mnist_dataset_train)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=BS, drop_last=False)
    train_loader = DataLoader(mnist_dataset_train, batch_size=1, batch_sampler=batch_sampler, num_workers=8)
    test_loader = DataLoader(mnist_dataset_test, batch_size=BS_test, shuffle=False)

    print('initialize network...')
    network = ClsNet()
    network.to(torch.device('cuda'))
    network = DistributedDataParallel(network, device_ids=[dist.get_rank()], find_unused_parameters=True)

    print('setup optimizer...')
    optimizer = optim.SGD(network.parameters(), lr=LR, momentum=MMT)
    for cur_epoch in range(EPOCH):
        network.train()
        for bid, (data, target) in enumerate(train_loader):
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            loss.backward()
            optimizer.step()
            if (bid+1) % log_internal == 0:
                print('Train Epoch:{} - batch_id:{} - Loss:{}'.format(cur_epoch+1, bid+1, loss.item()))
        torch.save(network.state_dict(), 'clsNet.pth')
        if dist.get_rank()==0:
            test(network, test_loader)
        dist.barrier()

def distributed_worker(local_rank, world_size):
    print(f'in distributed_worker....world_size={world_size}, local_rank={local_rank}')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8790'
    dist.init_process_group(backend='nccl', world_size=world_size, rank=local_rank)
    print('finish dist.init_process_group....')
    torch.cuda.set_device(local_rank)
    main_func()

'''Train Process
'''
def train(world_size):
    print('mp.spawn(....')
    if torch.cuda.is_available():
        mp.spawn(distributed_worker, nprocs=world_size, args=(world_size,))
    else:
        print('cuda is not available')

def test(network, test_loader):
    print('execute network evaluation...')
    network.eval()
    correct = 0
    losses = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.cuda()
            target = target.cuda()
            output = network(data)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            losses += loss.item()
            pred = output.data.max(1, keepdim=True)[1] # values & indices
            correct += pred.eq(target.data.view_as(pred)).sum()
    print("Accuracy: %.4f" % (correct.item()/len(test_loader.dataset)), "Loss: %.10f" % (losses/len(test_loader)))

def pred(network, gpu, img_path, trans):
    print('execute network evaluation...')
    network.eval()
    correct = 0
    losses = 0
    with torch.no_grad():
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = Image.fromarray(img, mode='L')
        data = trans(img)
        data = torch.unsqueeze(data, 0)
        if gpu:
            data = data.cuda()
        output = network(data)
        print(output)
        print(output.max(1)[1])

if __name__ == '__main__':
    gpus = '0,1,2,3,4,5,6,7'
    gpu_num = len(gpus.split(','))
    print(f'gpus num: {gpu_num}')
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    train(gpu_num)
 
    '''
    # test alone
    network = ClsNet()
    network.load_state_dict(torch.load('clsNet.pth'))
    network.cuda()
    #test(network, gpu=True)

    from PIL import Image
    import cv2
    import numpy as np
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    pred(network, True, 'imgs/1.pgm', trans)
    '''
