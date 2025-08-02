from config import cfg
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm, trange
import torch.nn.functional as F
from resnetCopy1 import ResNet18
import glob
    
class Program(nn.Module):
    def __init__(self, cfg, gpu):
        super(Program, self).__init__()
        self.cfg = cfg
        self.gpu = gpu
        self.init_net()
        self.init_mask()
        self.W = Parameter(torch.randn(self.M.shape), requires_grad=True)
        
    def init_net(self):
        if self.cfg.net == 'clean':
            self.net = ResNet18()
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            mean = mean[..., np.newaxis, np.newaxis]
            std = np.array([0.229, 0.224, 0.225],dtype=np.float32)
            std = std[..., np.newaxis, np.newaxis]
            checkpoint = torch.load(os.path.join(self.cfg.models_dir, 'clean_cifar10/clean_cifar_0000.pt'), map_location='cuda:0')
            self.net.load_state_dict(checkpoint)
            self.mean = Parameter(torch.from_numpy(mean), requires_grad=False)
            self.std = Parameter(torch.from_numpy(std), requires_grad=False)

            
        elif self.cfg.net == 'backdoor':
            self.net = ResNet18()
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            mean = mean[..., np.newaxis, np.newaxis]
            std = np.array([0.229, 0.224, 0.225],dtype=np.float32)
            std = std[..., np.newaxis, np.newaxis]
            print(self.cfg.net)
            checkpoint = torch.load(os.path.join(self.cfg.models_dir, 'badnet_cifar10/backdoor_cifar10_0015.pt'), map_location='cuda:0')
            self.net.load_state_dict(checkpoint)
            self.mean = Parameter(torch.from_numpy(mean), requires_grad=False)
            self.std = Parameter(torch.from_numpy(std), requires_grad=False)
            
            
        else:
            raise NotImplementedError()

        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
            
        for param in self.net.linear.parameters():
            param.requires_grad = True

        for param in self.net.layer4[1].bn2.parameters():
            param.requires_grad = True

        for param in self.net.layer4[1].bn1.parameters():
            param.requires_grad = True

        for param in self.net.layer4[0].bn2.parameters():
            param.requires_grad = True

        for param in self.net.layer4[0].bn1.parameters():
            param.requires_grad = True
            


    def init_mask(self):
        M = torch.ones(3, self.cfg.h1, self.cfg.w1)
        c_w, c_h = int(np.ceil(self.cfg.w1/2.)), int(np.ceil(self.cfg.h1/2.))
        M[:,c_h-self.cfg.h2//2:c_h+self.cfg.h2//2, c_w-self.cfg.w2//2:c_w+self.cfg.w2//2] = 0
        self.M = Parameter(M, requires_grad=False)

    def imagenet_label2_mnist_label(self, imagenet_label):
        return imagenet_label[:,:10]

    def forward(self, image):
        if self.cfg.dataset == 'mnist':
            image = image.repeat(1,3,1,1)
            
        elif self.cfg.dataset == 'cifar10':
            image = image.repeat(1,1,1,1)
            
        elif self.cfg.dataset == 'STL10':
            image = image.repeat(1,1,1,1)
            
        
        X = image.data.new(self.cfg.batch_size_per_gpu, 3, self.cfg.h1, self.cfg.w1)
        X[:] = 0
        X[:,:,int((self.cfg.h1-self.cfg.h2)//2):int((self.cfg.h1+self.cfg.h2)//2), int((self.cfg.w1-self.cfg.w2)//2):int((self.cfg.w1+self.cfg.w2)//2)] = image.data.clone()
        X = Variable(X, requires_grad=True)
        P = torch.sigmoid(self.W * self.M)
        X_adv = X + P
        X_adv = (X_adv - self.mean) / self.std
        Y_adv = self.net(X_adv)
        Y_adv = F.softmax(Y_adv, 1)
        return self.imagenet_label2_mnist_label(Y_adv)


class Adversarial_Reprogramming(object):
    def __init__(self, args, cfg=cfg):
        self.mode = args.mode
        self.gpu = args.gpu
        self.restore = args.restore
        self.cfg = cfg
        self.init_dataset()
        self.Program = Program(self.cfg, self.gpu)
        self.restore_from_file()
        self.set_mode_and_gpu()

    def init_dataset(self):
        seed = 42
        torch.manual_seed(seed)
        rng = torch.Generator()
        rng.manual_seed(seed)

        
        if self.cfg.dataset == 'mnist':
            train_set = torchvision.datasets.MNIST(os.path.join(self.cfg.data_dir, 'mnist'), train=True, transform=transforms.ToTensor(), download=True)
            test_set = torchvision.datasets.MNIST(os.path.join(self.cfg.data_dir, 'mnist'), train=False, transform=transforms.ToTensor(), download=True)
            #test_sampler = torch.utils.data.RandomSampler(test_set, num_samples=len(test_set), replacement=False, generator=torch.Generator().manual_seed(42))
            kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
            #indices = [i for i, (x, y) in enumerate(test_set) if y == 0]
            #test_set = torch.utils.data.Subset(test_set, indices)
            if self.gpu:
                self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu*len(self.gpu), shuffle=True,  **kwargs)
                self.test_loader = torch.utils.data.DataLoader(test_set, generator=rng, batch_size=self.cfg.batch_size_per_gpu*len(self.gpu),  **kwargs)
            else:
                self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu, shuffle=True,  **kwargs)
                self.test_loader = torch.utils.data.DataLoader(test_set, generator=rng, batch_size=self.cfg.batch_size_per_gpu,  **kwargs)
                
                
        elif self.cfg.dataset == 'cifar10':
            train_set = torchvision.datasets.CIFAR10(os.path.join(self.cfg.data_dir, 'cifar10'), train=True, transform=transforms.ToTensor(), download=True)
            test_set = torchvision.datasets.CIFAR10(os.path.join(self.cfg.data_dir, 'cifar10'), train=False, transform=transforms.ToTensor(), download=True)
            #test_sampler = torch.utils.data.RandomSampler(test_set, num_samples=len(test_set), replacement=False, generator=torch.Generator().manual_seed(42))
            kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
            #test_indices = np.where(np.array(test_set.targets) == 0)[0]
            #test_set = torch.utils.data.Subset(test_set, test_indices)
            if self.gpu:
                self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu*len(self.gpu), shuffle=True,  **kwargs)
                self.test_loader = torch.utils.data.DataLoader(test_set, generator=rng, batch_size=self.cfg.batch_size_per_gpu*len(self.gpu),  **kwargs)
            else:
                self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu, shuffle=True,  **kwargs)
                self.test_loader = torch.utils.data.DataLoader(test_set, generator=rng, batch_size=self.cfg.batch_size_per_gpu,  **kwargs)
                
        elif self.cfg.dataset == 'STL10':
            train_set = torchvision.datasets.STL10(os.path.join(self.cfg.data_dir, 'STL10'), split='train', transform=transforms.ToTensor(), download=True)
            test_set = torchvision.datasets.STL10(os.path.join(self.cfg.data_dir, 'STL10'), split='test', transform=transforms.ToTensor(),download=True)
            #test_sampler = torch.utils.data.RandomSampler(test_set, num_samples=len(test_set), replacement=False, generator=torch.Generator().manual_seed(42))
            kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
            #test_indices = np.where(np.array(test_set.targets) == 0)[0]
            #test_set = torch.utils.data.Subset(test_set, test_indices)
            if self.gpu:
                self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu*len(self.gpu), shuffle=True,  **kwargs)
                self.test_loader = torch.utils.data.DataLoader(test_set, generator=rng, batch_size=self.cfg.batch_size_per_gpu*len(self.gpu),  **kwargs)
            else:
                self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg.batch_size_per_gpu, shuffle=True,  **kwargs)
                self.test_loader = torch.utils.data.DataLoader(test_set, generator=rng, batch_size=self.cfg.batch_size_per_gpu,  **kwargs)
                
                
        else:
            raise NotImplementedError()

    def restore_from_file(self):
        if self.restore is not None:
            ckpt = os.path.join(self.cfg.train_dir, 'W_%03d.pt' % self.restore)
            assert os.path.exists(ckpt)
            if self.gpu:
                self.Program.load_state_dict(torch.load(ckpt), strict=False)
            else:
                self.Program.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
            self.start_epoch = self.restore + 1
        else:
            self.start_epoch = 1

    def set_mode_and_gpu(self):
        if self.mode == 'train':
            # optimizer
            self.BCE = torch.nn.BCELoss()
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.Program.parameters()), lr=self.cfg.lr, betas=(0.5, 0.999))
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=self.cfg.decay)

            if self.gpu:
                with torch.cuda.device(0):
                    self.BCE.cuda()
                    self.Program.cuda()

            if len(self.gpu) > 1:
                self.Program = torch.nn.DataParallel(self.Program, device_ids=list(range(len(self.gpu))))

        elif self.mode == 'validate' or self.mode == 'test':
            if self.gpu:
                with torch.cuda.device(0):
                    self.Program.cuda()

            if len(self.gpu) > 1:
                self.Program = torch.nn.DataParallel(self.Program, device_ids=list(range(len(self.gpu))))
        else:
            raise NotImplementedError()

    @property
    def get_W(self):
        for p in self.Program.parameters():
            if p.requires_grad:
                return p

    def imagenet_label2_mnist_label(self, imagenet_label):
        return imagenet_label[:,:10]

    def tensor2var(self, tensor, requires_grad=False, volatile=False):
        if self.gpu:
            with torch.cuda.device(0):
                tensor = tensor.cuda()
        return Variable(tensor, requires_grad=requires_grad, volatile=volatile)

    def compute_loss(self, out, label):
        if self.gpu:
            label = torch.zeros(self.cfg.batch_size_per_gpu*len(self.gpu), 10).scatter_(1, label.view(-1,1), 1)
        else:
            label = torch.zeros(self.cfg.batch_size_per_gpu, 10).scatter_(1, label.view(-1,1), 1)
        label = self.tensor2var(label)
        return self.BCE(out, label) + self.cfg.lmd * torch.norm(self.get_W) ** 2
    

    def validate(self):
        acc = 0.0
        processed_images = 0
        for k, (image, label) in enumerate(self.test_loader):
            image = self.tensor2var(image)
            out = self.Program(image)
            pred = out.data.cpu().numpy().argmax(1)
            acc += sum(label.numpy() == pred) / float(len(label) * len(self.test_loader))

        print('test accuracy: %.6f' % acc)

        
    def train(self):
        for self.epoch in range(self.start_epoch, self.cfg.max_epoch + 1):
            self.lr_scheduler.step()
            for j, (image, label) in tqdm(enumerate(self.train_loader)):
                if j > 2: break;
                #print(label)
                image = self.tensor2var(image)
                self.out = self.Program(image)
                self.loss = self.compute_loss(self.out, label)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
            print('epoch: %03d/%03d, loss: %.6f' % (self.epoch, self.cfg.max_epoch, self.loss.data.cpu().numpy()))
            torch.save(self.Program.net.state_dict(), os.path.join('train_log', 'W_%03d.pt' % self.epoch))
        self.validate()
            
    def test(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument('-r', '--restore', default=None, action='store', type=int, help='Specify checkpoint id to restore.')
    parser.add_argument('-g', '--gpu', default=[], nargs='+', type=str, help='Specify GPU ids.')
    # test params

    args = parser.parse_args()
    # print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    AR = Adversarial_Reprogramming(args)
    if args.mode == 'train':
        AR.train()
    elif args.mode == 'validate':
        AR.validate()
    elif args.mode == 'test':
        AR.test()
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()