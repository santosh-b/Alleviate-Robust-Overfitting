import os
import time 
import torch
import random
import shutil
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad
from advertorch.utils import NormalizeByChannelMeanStd
from datasets import *
from models.preactivate_resnet import *
from models.vgg import *
from models.wideresnet import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os, time
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from scipy.ndimage import filters
import torch as ch
from argparse import ArgumentParser
import os
import math
import sys
from torch.autograd import Variable

from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

# __all__ = ['save_checkpoint', 'setup_dataset_models', 'setup_dataset_models_standard', 'setup_seed', 'moving_average', 'bn_update', 'print_args',
#             'train_epoch', 'train_epoch_adv', 'train_epoch_adv_dual_teacher',
#             'test', 'test_adv']

def cifar10_dataloaders(batch_size=64, data_dir = 'datasets/cifar10'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def cifar100_dataloaders(batch_size=64, data_dir = 'datasets/cifar100'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def tiny_imagenet_dataloaders(batch_size=64, data_dir = 'datasets/tiny-imagenet-200', permutation_seed=10):

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'training')
    val_path = os.path.join(data_dir, 'validation')

    np.random.seed(permutation_seed)
    split_permutation = list(np.random.permutation(100000))

    train_set = Subset(ImageFolder(train_path, transform=train_transform), split_permutation[:90000])
    val_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[90000:])
    test_set = ImageFolder(val_path, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def save_checkpoint(state, is_SA_best, is_RA_best, is_SA_best_swa, is_RA_best_swa, save_path, filename='checkpoint.pth.tar'):
    
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)

    if is_SA_best_swa:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SWA_SA_best.pth.tar'))
    if is_RA_best_swa:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SWA_RA_best.pth.tar'))
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SA_best.pth.tar'))
    if is_RA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_RA_best.pth.tar'))

#print training configuration
def print_args(args):
    print('*'*50)
    print('Dataset: {}'.format(args.dataset))
    print('Model: {}'.format(args.arch))
    if args.arch == 'wideresnet':
        print('Depth {}'.format(args.depth_factor))
        print('Width {}'.format(args.width_factor))
    print('*'*50)        
    print('Attack Norm {}'.format(args.norm))  
    print('Test Epsilon {}'.format(args.test_eps))
    print('Test Steps {}'.format(args.test_step))
    print('Train Steps Size {}'.format(args.test_gamma))
    print('Test Randinit {}'.format(args.test_randinit))
    if args.eval:
        print('Evaluation')
        print('Loading weight {}'.format(args.pretrained))
    else:
        print('Training')
        print('Train Epsilon {}'.format(args.train_eps))
        print('Train Steps {}'.format(args.train_step))
        print('Train Steps Size {}'.format(args.train_gamma))
        print('Train Randinit {}'.format(args.train_randinit))
        print('SWA={0}, start point={1}, swa_c={2}'.format(args.swa, args.swa_start, args.swa_c_epochs))
        print('LWF={0}, coef_ce={1}, coef_kd1={2}, coef_kd2={3}, start={4}, end={5}'.format(
            args.lwf, args.coef_ce, args.coef_kd1, args.coef_kd2, args.lwf_start, args.lwf_end
        ))

# prepare dataset and models
def setup_dataset_models(args):

    # prepare dataset
    if args.dataset == 'cifar10':
        classes = 10
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    elif args.dataset == 'cifar100':
        classes = 100
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    elif args.dataset == 'tinyimagenet':
        classes = 200
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_loader, val_loader, test_loader = tiny_imagenet_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    else:
        raise ValueError("Unknown Dataset")

    #prepare model

    if args.arch == 'resnet18':
        model = ResNet18(num_classes = classes)
        model.normalize = dataset_normalization

        if args.swa:
            swa_model = ResNet18(num_classes = classes)
            swa_model.normalize = dataset_normalization
        else:
            swa_model = None

        if args.lwf:
            teacher1 = ResNet18(num_classes = classes)
            teacher1.normalize = dataset_normalization
            teacher2 = ResNet18(num_classes = classes)
            teacher2.normalize = dataset_normalization
        else:
            teacher1 = None
            teacher2 = None 

    elif args.arch == 'wideresnet':
        model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
        model.normalize = dataset_normalization

        if args.swa:
            swa_model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
            swa_model.normalize = dataset_normalization
        else:
            swa_model = None

        if args.lwf:
            teacher1 = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
            teacher1.normalize = dataset_normalization
            teacher2 = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
            teacher2.normalize = dataset_normalization
        else:
            teacher1 = None
            teacher2 = None 

    elif args.arch == 'vgg16':
        model = vgg16_bn(num_classes = 10)
        model.normalize = dataset_normalization

        if args.swa:
            swa_model = vgg16_bn(num_classes = 10)
            swa_model.normalize = dataset_normalization
        else:
            swa_model = None

        if args.lwf:
            teacher1 = vgg16_bn(num_classes = 10)
            teacher1.normalize = dataset_normalization
            teacher2 = vgg16_bn(num_classes = 10)
            teacher2.normalize = dataset_normalization
        else:
            teacher1 = None
            teacher2 = None 

    else:
        raise ValueError("Unknown Model")   
    
    return train_loader, val_loader, test_loader, model, swa_model, teacher1, teacher2

def setup_dataset_models_standard(args):

    # prepare dataset
    if args.dataset == 'cifar10':
        classes = 10
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    elif args.dataset == 'cifar100':
        classes = 100
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    elif args.dataset == 'tinyimagenet':
        classes = 200
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_loader, val_loader, test_loader = tiny_imagenet_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
    else:
        raise ValueError("Unknown Dataset")

    #prepare model

    if args.arch == 'resnet18':
        model = ResNet18(num_classes = classes)
        model.normalize = dataset_normalization

    elif args.arch == 'wideresnet':
        model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
        model.normalize = dataset_normalization

    elif args.arch == 'vgg16':
        model = vgg16_bn(num_classes = 10)
        model.normalize = dataset_normalization

    else:
        raise ValueError("Unknown Model")   
    
    return train_loader, val_loader, test_loader, model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

# knowledge distillation loss function
def loss_fn_kd(scores, target_scores, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    if not scores.size(1) == target_scores.size(1):
        print('size does not match')

    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2

    return KD_loss

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


# training 
def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    start = time.time()
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(input)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def train_epoch_adv(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    model.train()
    start = time.time()
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        with ctx_noparamgrad(model):
            input_adv = adversary.perturb(input, target)

        # compute output
        output_adv = model(input_adv)
        loss = criterion(output_adv, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def train_epoch_adv_dual_teacher(train_loader, model, teacher1, teacher2, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    model.train()
    teacher1.eval()
    teacher2.eval()
    start = time.time()
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        with ctx_noparamgrad(model):
            input_adv = adversary.perturb(input, target)

        # compute output
        output_adv = model(input_adv)

        with torch.no_grad():
            target_score1 = teacher1(input_adv)
            target_score2 = teacher2(input_adv)

        loss_KD = loss_fn_kd(output_adv, target_score1, T=args.temperature)
        loss_KD2 = loss_fn_kd(output_adv, target_score2, T=args.temperature)

        loss = criterion(output_adv, target)*args.coef_ce + loss_KD*args.coef_kd1 + loss_KD2*args.coef_kd2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

#testing
def test(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    start = time.time()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()
    
        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('Standard Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def test_adv(val_loader, model, criterion, args):
    """
    Run adversarial evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    model.eval()
    start = time.time()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        input_adv = adversary.perturb(input, target)
        # compute output
        with torch.no_grad():
            output = model(input_adv)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('Robust Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

class EarlyBird():
    def __init__(self, percent, epoch_keep=5):
        self.percent = percent
        self.epoch_keep = epoch_keep
        self.masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]

    def pruning(self, model, percent):
        total = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * percent)
        thre = y[thre_index]
        # print('Pruning threshold: {}'.format(thre))

        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
                index += size

        # print('Pre-processing Successful!')
        return mask

    def put(self, mask):
        if len(self.masks) < self.epoch_keep:
            self.masks.append(mask)
        else:
            self.masks.pop(0)
            self.masks.append(mask)

    def cal_dist(self):
        if len(self.masks) == self.epoch_keep:
            for i in range(len(self.masks)-1):
                mask_i = self.masks[-1]
                mask_j = self.masks[i]
                self.dists[i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0)
            return True
        else:
            return False

    def early_bird_emerge(self, model):
        mask = self.pruning(model, self.percent)
        self.put(mask)
        flag = self.cal_dist()
        if flag == True:
            print(self.dists)
            for i in range(len(self.dists)):
                if self.dists[i] > 0.1:
                    return False
            return True
        else:
            return False

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class vgg(nn.Module):
    def __init__(self, depth, dataset='cifar10', init_weights=True, cfg=None, seed=0):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'imagenet':
            num_classes = 1000
        self.classifier = nn.Linear(cfg[-1], num_classes)
        rng = torch.manual_seed(seed)
        if init_weights:
            self._initialize_weights(rng)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not fake_relu) and (not no_relu),  \
            "fake_relu and no_relu not yet supported for this architecture"
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        latent = x.view(x.size(0), -1)
        y = self.classifier(latent)
        if with_latent:
            return y, latent
        return y

    def _initialize_weights(self, rng):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n), generator=rng)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01, generator=rng)
                m.bias.data.zero_()

def prune(model, pct):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    # comparsion and permutation (sort process)
    p_flops += total * np.log2(total) * 3
    thre_index = int(total * pct)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre.cuda()).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            if int(torch.sum(mask)) > 0:
                cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    return cfg

def resprune(model, pct):
    total = 0

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    p_flops += total * np.log2(total) * 3
    thre_index = int(total * pct)
    thre = y[thre_index]


    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre.cuda()).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            num = int(torch.sum(mask))
            if num != 0:
                cfg.append(num)
                cfg_mask.append(mask.clone())
            elif num == 0:
                cfg.append(1)
                _mask = mask.clone()
                _mask[0] = 1
                cfg_mask.append(_mask)
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            # cfg.append('M')
            pass

    pruned_ratio = pruned/total
    #print(pruned_ratio)
    return cfg

def fix_robustness_ckpt(ckpt):
    state_dict = {}
    try:
        for key in ckpt['model']:
            cleaned = key.replace('module.model.','')
            state_dict[cleaned] = ckpt['model'][key]
    except:
        for key in ckpt:
            cleaned = key.replace('module.model.','')
            state_dict[cleaned] = ckpt[key]
    return state_dict

def get_pruned_init(model, cfg, pct, dataset):
    modelnew = vgg(16, dataset=dataset, cfg=cfg)
    model.cuda()
    modelnew.cuda()

    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    # comparsion and permutation (sort process)
    p_flops += total * np.log2(total) * 3
    thre_index = int(total * pct)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre.cuda()).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            if int(torch.sum(mask)) > 0:
                cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    model = vgg(16, dataset=dataset, seed=0)
    model.cuda()

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), modelnew.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            if torch.sum(end_mask) == 0:
                continue
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if torch.sum(end_mask) == 0:
                continue
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # random set for test
            # new_end_mask = np.asarray(end_mask.cpu().numpy())
            # new_end_mask = np.append(new_end_mask[int(len(new_end_mask)/2):], new_end_mask[:int(len(new_end_mask)/2)])
            # idx1 = np.squeeze(np.argwhere(new_end_mask))

            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    return modelnew, cfg_mask

def get_resnet_pruned_init(model, cfg, pct, dataset):
    if dataset == 'cifar10':
        modelnew = resnet18(seed=0, num_classes=10, cfg=cfg)
        model.cuda()
        modelnew.cuda()
    elif dataset == 'cifar100':
        modelnew = resnet18(seed=0, num_classes=100, cfg=cfg)
        model.cuda()
        modelnew.cuda()

    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    # comparsion and permutation (sort process)
    p_flops += total * np.log2(total) * 3
    thre_index = int(total * pct)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre.cuda()).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            num = int(torch.sum(mask))
            if num != 0:
                cfg.append(num)
                cfg_mask.append(mask.clone())
            elif num == 0:
                cfg.append(1)
                _mask = mask.clone()
                _mask[0] = 1
                cfg_mask.append(_mask)
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            # cfg.append('M')
            pass

    if dataset == 'cifar10':
        model = resnet18(seed=0, num_classes=10)
    elif dataset == 'cifar100':
        model = resnet18(seed=0, num_classes=100)
    model.cuda()

    old_modules = list(model.modules())
    new_modules = list(modelnew.modules())

    useful_i = []
    for i, module in enumerate(old_modules):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ReLU) or isinstance(module, channel_selection):
            useful_i.append(i)
    temp = []
    for i, item in enumerate(useful_i):
        temp.append(old_modules[item])
    # for i, item in enumerate(temp):
    #     print(i, item)
    # sys.exit()

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0
    bn_count = 0
    
    downsample = [8, 13, 18]
    last_block = [3, 5, 7, 10, 12, 15, 17, 20]
    for layer_id in range(len(temp)):
        m0 = old_modules[useful_i[layer_id]]
        m1 = new_modules[useful_i[layer_id]]
        # print(m0)
        if isinstance(m0, nn.BatchNorm2d):
            bn_count += 1
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print(layer_id_in_cfg, len(cfg_mask))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if bn_count == 1:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                m2 = new_modules[useful_i[layer_id+2]] # channel selection
                assert isinstance(m2, channel_selection)
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]

            elif bn_count in downsample:
                # If the current layer is the downsample layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

            elif bn_count in last_block:
                # If the current layer is the last conv-bn layer in block, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                if bn_count + 1 in downsample:
                    m2 = new_modules[useful_i[layer_id+3]]
                    assert isinstance(m2, channel_selection)
                else:
                    m2 = new_modules[useful_i[layer_id+1]]
                    assert isinstance(m2, channel_selection) or isinstance(m2, nn.Linear)
                if isinstance(m2, channel_selection):
                    m2.indexes.data.zero_()
                    m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            conv_count += 1
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                continue
            elif conv_count in downsample: # downsample
                # We need to consider the case where there are downsampling convolutions.
                # For these convolutions, we just copy the weights.
                m1.weight.data = m0.weight.data.clone()
                continue
            elif conv_count in last_block:
                # the last convolution in the residual block.
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
                continue
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            # end_mask = cfg_mask[-1]
            # idx0 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print(idx0)
            # if idx0.size == 1:
            #     idx0 = np.resize(idx0, (1,))
            # m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

    return modelnew, cfg_mask


## resnet definition

class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
	    """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
		"""
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,)) 
        output = input_tensor[:, selected_index, :, :]
        return output

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.select = channel_selection(inplanes)
        self.conv1 = conv3x3(cfg[0], cfg[1], stride)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg[1], planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.select(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, seed, num_classes=10, cfg=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        rng = torch.manual_seed(seed)
        if cfg is None:
            # Construct config variable (basic block)
            cfg = [[64], [64, 64]*layers[0], [128, 128]*layers[1], [256, 256]*layers[2], [512, 512]* layers[3]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], cfg=cfg[0:2*layers[0]])
        self.layer2 = self._make_layer(block, 128, layers[1], cfg=cfg[2*layers[0]: 2*(layers[0]+layers[1])], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg=cfg[2*(layers[0]+layers[1]): 2*(layers[0]+layers[1]+layers[2])], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg=cfg[2*(layers[0]+layers[1]+layers[2]): 2*(layers[0]+layers[1]+layers[2]+layers[3])], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.select = channel_selection(64 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n), generator=rng)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01, generator=rng)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:2], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[2*i: 2*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not fake_relu) and (not no_relu),  \
            "fake_relu and no_relu not yet supported for this architecture"
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        latent = x.view(x.size(0), -1)
        y = self.fc(latent)
        if with_latent:
            return y, latent
        return y


def resnet18(pretrained=False, cfg=None, num_classes=10, seed=0, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], seed, num_classes=num_classes, cfg=cfg, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet34'])))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet50'])))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet101'])))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet152'])))
    return model

class ResNet50(nn.Module):

    def __init__(self, block, layers, cfg, seed=0, num_classes=1000):
        self.inplanes = 64
        rng = torch.manual_seed(seed)
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, cfg[1:10], 64, layers[0])
        self.layer2 = self._make_layer(block, cfg[10:22], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[22:40], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, cfg[40:49], 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n), generator=rng)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01, generator=rng)
                m.bias.data.zero_()

    def _make_layer(self, block, cfg, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i:3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not fake_relu) and (not no_relu),  \
            "fake_relu and no_relu not yet supported for this architecture"
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        latent = x.view(x.size(0), -1)
        y = self.fc(latent)
        if with_latent:
            return y, latent
        return y

cfg_official = [[64, 64, 64], [256, 64, 64] * 2, [256, 128, 128], [512, 128, 128] * 3, 
                [512, 256, 256], [1024, 256, 256] * 5, [1024, 512, 512], [2048, 512, 512] * 2]
cfg_official = [item for sublist in cfg_official for item in sublist]
assert len(cfg_official) == 48, "Length of cfg_official is not right"


def resnet50_official(depth=50, cfg=None, pretrained=False, num_classes=10, seed=0):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if cfg == None:
        cfg_official = [[64], [64, 64, 64], [256, 64, 64] * 2, [256, 128, 128], [512, 128, 128] * 3, 
                    [512, 256, 256], [1024, 256, 256] * 5, [1024, 512, 512], [2048, 512, 512] * 2]
        cfg_official = [item for sublist in cfg_official for item in sublist]
        assert len(cfg_official) == 49, "Length of cfg_official is not right"
        cfg = cfg_official
    model = ResNet50(Bottleneck, [3, 4, 6, 3], cfg, num_classes=num_classes, seed=seed)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def get_resnet50_pruned_init(model, cfg, pct, dataset):
    if dataset == 'cifar10':
        modelnew = resnet50_official(seed=0, num_classes=10, cfg=cfg)
        model.cuda()
        modelnew.cuda()
    elif dataset == 'cifar100':
        modelnew = resnet50_official(seed=0, num_classes=100, cfg=cfg)
        model.cuda()
        modelnew.cuda()

    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    # comparsion and permutation (sort process)
    p_flops += total * np.log2(total) * 3
    thre_index = int(total * pct)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre.cuda()).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            num = int(torch.sum(mask))
            if num != 0:
                cfg.append(num)
                cfg_mask.append(mask.clone())
            elif num == 0:
                cfg.append(1)
                _mask = mask.clone()
                _mask[0] = 1
                cfg_mask.append(_mask)
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            # cfg.append('M')
            pass

    if dataset == 'cifar10':
        model = resnet50_official(seed=0, num_classes=10)
    elif dataset == 'cifar100':
        model = resnet50_official(seed=0, num_classes=100)
    model.cuda()

    old_modules = list(model.modules())
    new_modules = list(modelnew.modules())

    useful_i = []
    for i, module in enumerate(old_modules):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ReLU) or isinstance(module, channel_selection):
            useful_i.append(i)
    temp = []
    for i, item in enumerate(useful_i):
        temp.append(old_modules[item])
    # for i, item in enumerate(temp):
    #     print(i, item)
    # sys.exit()

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0
    bn_count = 0
    
    downsample = [5, 15, 28, 47] #[11, 37, 71, 121]
    last_block = [4, 8, 11, 14, 18, 21, 24, 27, 31, 34, 37, 40, 43, 46, 50, 53] #[8, 18, 26, 34, 44, 52, 60, 68, 78, 86, 94, 102, 110, 118, 128, 136]
    channel_select = [5, 15, 23, 31, 41, 49, 57, 65, 75, 83, 91, 99, 107, 115, 125, 133, 140]
    channel_select_bn = [1, 4, 8, 11, 14, 18, 21, 24, 27, 31, 34, 37, 40, 43, 46, 50, 53]#[1, 9, 19, 27, 35, 45, 53, 61, 69, 79, 87, 95, 103, 111, 119, 129, 137]
    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                if conv_count % 3 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions.
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    return modelnew, cfg_mask

