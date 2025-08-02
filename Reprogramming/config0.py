import os
from easydict import EasyDict

cfg = EasyDict()

cfg.net = 'backdoor'
cfg.dataset = 'mnist'

cfg.train_dir = 'train_log'
cfg.models_dir = 'models'
cfg.data_dir = 'datasets'

cfg.batch_size_per_gpu = 100
if cfg.dataset == 'mnist':
    cfg.w1 = 64
    cfg.h1 = 64
    cfg.w2 = 28
    cfg.h2 = 28
    
elif cfg.dataset == 'cifar10':
    cfg.w1 = 96
    cfg.h1 = 96
    cfg.w2 = 32
    cfg.h2 = 32
    
elif cfg.dataset == 'STL10':
    cfg.w1 = 148
    cfg.h1 = 148
    cfg.w2 = 96
    cfg.h2 = 96

    

cfg.lmd = 5e-7
cfg.lr = 0.05
cfg.decay = 0.96
cfg.max_epoch = 300

if not os.path.exists(cfg.train_dir):
    os.makedirs(cfg.train_dir)

if not os.path.exists(cfg.models_dir):
    os.makedirs(cfg.models_dir)

if not os.path.exists(cfg.data_dir):
    os.makedirs(cfg.data_dir)
