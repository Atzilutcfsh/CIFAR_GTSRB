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
import torchvision.models as models
import trojanvision
from resnet import ResNet 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
    
class Program(nn.Module):
    def __init__(self, cfg, gpu):
        super(Program, self).__init__()
        self.cfg = cfg
        self.gpu = gpu
        self.init_mask()
        self.W = Parameter(torch.randn(self.M.shape), requires_grad=True)
        
    def init_net(self, i):
        if self.cfg.net == 'clean':
            self.net = ResNet18()
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            mean = mean[..., np.newaxis, np.newaxis]
            std = np.array([0.229, 0.224, 0.225],dtype=np.float32)
            std = std[..., np.newaxis, np.newaxis]
            self.filename = '{:04d}'.format(i)
            print(self.filename)
            print(self.cfg.net)
            checkpoint = torch.load(os.path.join(self.cfg.models_dir, 'clean_cifar10', 'clean_cifar_' + self.filename + '.pt'), map_location='cuda:0')
            #checkpoint = torch.load(os.path.join(self.cfg.models_dir, 'clean_gt', 'cleangt_' + self.filename + '.pt'), map_location='cuda:0')
            self.net.load_state_dict(checkpoint)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.mean = Parameter(torch.from_numpy(mean).to(device), requires_grad=False)
            self.std = Parameter(torch.from_numpy(std).to(device), requires_grad=False)

            
        elif self.cfg.net == 'backdoor':
            self.net = ResNet18()
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            mean = mean[..., np.newaxis, np.newaxis]
            std = np.array([0.229, 0.224, 0.225],dtype=np.float32)
            std = std[..., np.newaxis, np.newaxis]
            self.filename = '{:04d}'.format(i)
            print(self.filename)
            print(self.cfg.net)
            checkpoint = torch.load(os.path.join(self.cfg.models_dir, 'badnet_cifar10', 'backdoor_cifar10_' + self.filename + '.pt'), map_location='cuda:0')
            #checkpoint = torch.load(os.path.join(self.cfg.models_dir, 'badnet_gt_0', 'badnet_gt_' + self.filename + '.pt'), map_location='cuda:0')
            self.net.load_state_dict(checkpoint, strict=False)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.mean = Parameter(torch.from_numpy(mean).to(device), requires_grad=False)
            self.std = Parameter(torch.from_numpy(std).to(device), requires_grad=False)


            
        else:
            raise NotImplementedError()

        self.net.to(torch.device('cuda'))
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False


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
        data_transform = transforms.Compose([
                #transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
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

            train_set = torchvision.datasets.CIFAR10(os.path.join(self.cfg.data_dir, 'cifar10'), train=True, transform=data_transform, download=True)
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
            train_set = torchvision.datasets.STL10(os.path.join(self.cfg.data_dir, 'STL10'), split='train', transform=data_transform ,download=True)
            test_set = torchvision.datasets.STL10(os.path.join(self.cfg.data_dir, 'STL10'), split='test', transform=data_transform,download=True)
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
    
    def hook_fn(self, module, input, output):
        global conv_output
        conv_output = output
        #return conv_output
    
    def validate(self):
        acc = 0.0
        processed_images = 0
        
        df = pd.read_csv('./new_stl_badnet.csv')
        data = df.loc[:, df.columns != 'target'].values
        target = df['target'].values

        # 数据预处理
        data = torch.from_numpy(data)
        data = data.reshape(-1, 3, 96, 96)
        #print(data.shape)
        target = np.where(target == 'target', 0, target)
        target = target.astype(int)
        target = torch.from_numpy(target).long()

        # 替换原始测试迭代
        device = torch.device("cuda")
        images = data.float().to(device)
        labels = target.to(device)
        #index = [] 
        conv_layer = self.Program.net.layer4[1].conv2
        handle = conv_layer.register_forward_hook(self.hook_fn)
        #for i in range(len(images)):
        prdict_y = []
        conv_outputs = []
        for i in range(len(images)):
            out = self.Program(images[i])
            y = out.argmax(1)
            #conv_output = handle.remove()
            #print(conv_output)
            conv_outputs.append(conv_output.cpu().detach().numpy())
            prdict_y.append(y.cpu().detach().numpy())
        prdict_y = np.vstack(prdict_y)
        prdict_y = torch.from_numpy(prdict_y)        
        conv_outputs = np.vstack(conv_outputs)
        conv_outputs = torch.from_numpy(conv_outputs)
        n,c,w,h = conv_outputs.shape
        conv_outputs = conv_outputs.view (n, -1)
        label = labels.cpu().numpy()
        print(conv_outputs.shape)
        # 進行t-SNE
        tsne = TSNE(n_components=2) 
        conv_output_2d = tsne.fit_transform(conv_outputs)

        # 畫圖 
        #plt.figure(figsize=(10, 10))
        #plt.scatter(conv_output_2d[:6499, 0], conv_output_2d[:6499, 1], c=label[:6499], cmap='tab10')
        #custom_colors = ['khaki'] * (conv_output_2d.shape[0] - 6499)
        #plt.scatter(conv_output_2d[6499:, 0], conv_output_2d[6499:, 1], c=custom_colors, marker='*')

        #plt.title('t-SNE Visualization of Adaptive_Patch Model')
        #plt.savefig('tsne_stl10_test_adp.pdf', format = 'pdf', dpi = 300)
        #plt.colorbar()
                #index.append(i)
        plt.figure(figsize=(16, 12))

        x_min, x_max = conv_output_2d[:, 0].min() - 1, conv_output_2d[:, 0].max() + 1
        y_min, y_max = conv_output_2d[:, 1].min() - 1, conv_output_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))

        grid_points_2d = np.c_[xx.ravel(), yy.ravel()]

        # For each point in the meshgrid, find the closest point in X_embedded
        grid_points_original_indices = np.argmin(np.linalg.norm(conv_output_2d[:, np.newaxis] - grid_points_2d, axis=2), axis=0)

        #prdict_y = net(images).argmax(1)
        # prdict_y = prdict_y.argmax(1)

        Z = prdict_y[grid_points_original_indices]
        Z = Z.reshape(xx.shape)

        grid_points_original = images[grid_points_original_indices]

        # Evaluate the decision_function on the approximated original points
        #Z = svm_rbf.decision_function(grid_points_original).argmax(1)
        #Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, levels=10, c=label, alpha=0.8)


        plt.scatter(conv_output_2d[:, 0], conv_output_2d[:, 1], c=label, marker='o', s=80, edgecolors='k')
        for i, txt in enumerate(images):
            plt.annotate(str(i), (conv_output_2d[i, 0], conv_output_2d[i, 1]))
        plt.savefig('tsne_stl10_badnet_boud.pdf', format = 'pdf', dpi = 300)
                #print('test accuracy: %.6f' % acc)

        
    def train(self):
        for i in range(1):
            self.Program.init_net(i)
            for self.epoch in range(self.start_epoch, self.cfg.max_epoch + 1):
                self.lr_scheduler.step()
                for j, (image, label) in enumerate(self.test_loader):
                    #if j > 1: break;
                    image = self.tensor2var(image)
                    self.out = self.Program(image)
                    self.loss = self.compute_loss(self.out, label)
                    self.optimizer.zero_grad()
                    self.loss.backward()
                    self.optimizer.step()
                print('net: %d, epoch: %03d/%03d, loss: %.6f' % (i, self.epoch, self.cfg.max_epoch, self.loss.data.cpu().numpy()))
                torch.save({'W': self.get_W}, os.path.join(self.cfg.train_dir, 'W_net%d_%03d.pt' % (i, self.epoch)))

                if self.epoch == self.cfg.max_epoch:
                    self.validate()
            #self.lr_scheduler.step()
            self.optimizer.param_groups[0]['lr'] = self.cfg.lr
            
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