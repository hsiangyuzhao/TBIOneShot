import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from abc import ABC, abstractmethod
from utils.iteration.iterator import MetricMeter


class BaseSegmentationModel(ABC):
    """
    This class is an abstract base class (ABC) for segmentation models.
    To create a subclass, you need to implement the following four methods:
       -- <__init__>:                      initialize the class.
       -- <set_input>:                     unpack data from dataset.
       -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
       -- <evaluate_one_step>:                      performance evaluation.
    """
    def __init__(self, cfg, num_classes, amp=False):
        # initialize training CUDA devices
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # training configuration
        self.cfg = cfg
        self.num_classes = num_classes
        self.is_mixed = amp
        self.scaler = GradScaler()
        self.start_epoch = -1

        # initialize networks, criterion, optimizer and scheduler
        self.network = None
        self.seg_criterion = None
        self.optimizer = None
        self.scheduler = None

        # visualization
        self.visual_names = []
        self.loss_names = []

    def one_hot(self, inputs):
        # inputs shape: B, 1, H, W, D
        return F.one_hot(inputs.squeeze(1).long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)

    def train(self):
        self.network.train()
        return self

    def eval(self):
        self.network.eval()
        return self

    def training(self):
        return self.network.training

    def initialize_metric_meter(self, class_list):
        self.class_list = class_list
        self.metric_meter = MetricMeter(metrics=['dice', 'hd95', 'asd'], class_names=class_list, subject_names=['name'])
        self.train_loss = MetricMeter(metrics=self.loss_names, class_names=['train'])
        self.val_loss = MetricMeter(metrics=['loss'], class_names=['val'])

    def update_loss_meter(self, print=False):
        loss_dict = {}
        for loss_name in self.loss_names:
            try:
                loss_value = float(getattr(self, loss_name))
            except:
                continue
            loss_dict['train_{}'.format(loss_name)] = loss_value
        self.train_loss.update(loss_dict)
        stats = self.train_loss.report(print_stats=print, mean_only=True)
        return stats

    @abstractmethod
    def set_input(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def optimize_parameters(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate_one_step(self, *args, **kwargs):
        raise NotImplementedError

    def load_networks(self, ckpt_path, resume_training=False):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        print('Load ckpt weight: {}'.format(ckpt_path))
        self.network.load_state_dict(checkpoint['net'])
        if resume_training:
            print('Load training config for breakpoint continuation')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.start_epoch = checkpoint['epoch']

    def save_networks(self, epoch_index, save_dir):
        if dist.get_rank() == 0:
            checkpoint = {
                "net": self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
                "epoch": epoch_index
            }
            torch.save(checkpoint,
                       os.path.join(save_dir, 'Epoch_{}.pkl'.format(epoch_index)))


class MultiNetworkSegmentationModel(ABC):
    """
    This class is an abstract base class (ABC) for segmentation models.
    To create a subclass, you need to implement the following four methods:
       -- <__init__>:                      initialize the class.
       -- <set_input>:                     unpack data from dataset.
       -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
       -- <evaluate_one_step>:                      performance evaluation.
    """
    def __init__(self, cfg, num_classes, amp=False):
        # initialize training CUDA devices
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # training configuration
        self.cfg = cfg
        self.num_classes = num_classes
        self.is_mixed = amp
        self.scaler = GradScaler()
        self.start_epoch = -1

        # initialize networks, criterion, optimizer and scheduler
        self.net_names = []

        # visualization
        self.visual_names = []
        self.loss_names = []

    def one_hot(self, inputs):
        # inputs shape: B, 1, H, W, D
        return F.one_hot(inputs.squeeze(1).long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

    def train(self):
        for name in self.net_names:
            net = getattr(self, name)
            net.train()
        return self

    def eval(self):
        for name in self.net_names:
            net = getattr(self, name)
            net.eval()
        return self

    def training(self):
        return getattr(self, self.net_names[0]).training

    def initialize_metric_meter(self, class_list):
        self.class_list = class_list
        self.metric_meter = MetricMeter(metrics=['dice', 'hd95', 'asd'], class_names=class_list, subject_names=['name'])
        self.train_loss = MetricMeter(metrics=self.loss_names, class_names=['train'])
        self.val_loss = MetricMeter(metrics=['loss'], class_names=['val'])

    def update_loss_meter(self, print=False):
        loss_dict = {}
        for loss_name in self.loss_names:
            try:
                loss_value = float(getattr(self, loss_name))
            except:
                continue
            loss_dict['train_{}'.format(loss_name)] = loss_value
        self.train_loss.update(loss_dict)
        stats = self.train_loss.report(print_stats=print, mean_only=True)
        return stats

    @abstractmethod
    def set_input(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def optimize_parameters(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate_one_step(self, *args, **kwargs):
        raise NotImplementedError

    def load_networks(self, ckpt_path, resume_training=False):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        print('Load ckpt weight: {}'.format(ckpt_path))
        if resume_training:
            print('Load training config for breakpoint continuation')
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.start_epoch = checkpoint['epoch']
        for name in self.net_names:
            getattr(self, name).load_state_dict(checkpoint[name])
            if resume_training:
                getattr(self, '{}_optimizer'.format(name)).load_state_dict(checkpoint['{}_optimizer'.format(name)])
                getattr(self, '{}_scheduler'.format(name)).load_state_dict(checkpoint['{}_scheduler'.format(name)])

    def load_single_network(self, ckpt_path, net_name, resume_training=False):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        print('Load ckpt weight: {}'.format(ckpt_path))
        if resume_training:
            print('Load training config for breakpoint continuation')
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.start_epoch = checkpoint['epoch']
        getattr(self, net_name).load_state_dict(checkpoint[net_name])
        if resume_training:
            getattr(self, '{}_optimizer'.format(net_name)).load_state_dict(checkpoint['{}_optimizer'.format(net_name)])
            getattr(self, '{}_scheduler'.format(net_name)).load_state_dict(checkpoint['{}_scheduler'.format(net_name)])

    def save_networks(self, epoch_index, save_dir):
        if dist.get_rank() == 0:
            checkpoint = {}
            for name in self.net_names:
                checkpoint[name] = getattr(self, name).state_dict()
                checkpoint['{}_optimizer'.format(name)] = getattr(self, '{}_optimizer'.format(name)).state_dict()
                checkpoint['{}_scheduler'.format(name)] = getattr(self, '{}_scheduler'.format(name)).state_dict()
            checkpoint['scaler'] = self.scaler.state_dict()
            checkpoint['epoch'] = epoch_index
            torch.save(checkpoint, os.path.join(save_dir, 'Epoch_{}.pkl'.format(epoch_index)))

    def save_best_networks(self, epoch_index, save_dir):
        if dist.get_rank() == 0:
            checkpoint = {}
            for name in self.net_names:
                checkpoint[name] = getattr(self, name).state_dict()
                checkpoint['{}_optimizer'.format(name)] = getattr(self, '{}_optimizer'.format(name)).state_dict()
                checkpoint['{}_scheduler'.format(name)] = getattr(self, '{}_scheduler'.format(name)).state_dict()
            checkpoint['scaler'] = self.scaler.state_dict()
            checkpoint['epoch'] = epoch_index
            torch.save(checkpoint, os.path.join(save_dir, 'Epoch_best.pkl'))
