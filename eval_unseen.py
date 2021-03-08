from utils import *
import torch

import os
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms


import torch.nn.functional as F
import inspect
import torch.nn as nn

unpruned_eb = sys.argv[1]
final_weights = sys.argv[2]
pct = sys.argv[3]
log_folder = sys.argv[4]
dataset = sys.argv[5]
model_type = sys.argv[6]
aa_eval = sys.argv[7]

aa_eval = True if aa_eval == 'True' else False
is_pruned = True if unpruned_eb != final_weights else False
pct = float(pct)

# =============================================== Tickets ==============================================================
if is_pruned:
    weight_before_prune = fix_robustness_ckpt(torch.load(unpruned_eb))
    if model_type == 'resnet18':
        if dataset == 'cifar10':
            model = resnet18(seed=0, num_classes=10)
            model.load_state_dict(weight_before_prune, strict=False)
            cfg = resprune(model.cuda(), pct)
        elif dataset == 'cifar100':
            model = resnet18(seed=0, num_classes=100)
            model.load_state_dict(weight_before_prune, strict=False)
            cfg = resprune(model.cuda(), pct)
            os.system('pip install cifar2png')
            os.system('cifar2png cifar100 cifar100')
        elif dataset == 'tiny':
            model = resnet18(seed=0, num_classes=200)
            model.load_state_dict(weight_before_prune, strict=False)
            cfg = resprune(model.cuda(), pct)

    if model_type == 'resnet50':
        if dataset == 'cifar10':
            model = resnet50_official(seed=0, num_classes=10)
            model.load_state_dict(weight_before_prune, strict=False)
            cfg = resprune(model.cuda(), pct)
        elif dataset == 'cifar100':
            model = resnet50_official(seed=0, num_classes=100)
            model.load_state_dict(weight_before_prune, strict=False)
            cfg = resprune(model.cuda(), pct)
            os.system('pip install cifar2png')
            os.system('cifar2png cifar100 cifar100')
        elif dataset == 'tiny':
            model = resnet50_official(seed=0, num_classes=200)
            model.load_state_dict(weight_before_prune, strict=False)
            cfg = resprune(model.cuda(), pct)

    if model_type == 'vgg16':
        if dataset == 'cifar10':
            model = vgg(16, seed=0, dataset='cifar10')
            model.load_state_dict(weight_before_prune)
            cfg = vggprune(model.cuda(), pct)
        elif dataset == 'cifar100':
            model = vgg(16, seed=0, dataset='cifar10')
            model.load_state_dict(weight_before_prune)
            cfg = vggprune(model.cuda(), pct)
else:
    cfg=None

# =============================================== Dataset / Model ======================================================
transform_list = [transforms.ToTensor()]
transform_chain = transforms.Compose(transform_list)
print('DATASET',dataset)
if model_type == 'resnet50':
    if dataset == 'cifar10':
        model = resnet50_official(num_classes=10, cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        item = datasets.CIFAR10(root='cifar10', train=False, transform=transform_chain, download=True)
        _, _, test_loader = cifar10_dataloaders(data_dir='cifar10')
    elif dataset == 'cifar100':
        model = resnet50_official(num_classes=100, cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        item = datasets.CIFAR100(root='cifar100', train=False, transform=transform_chain, download=True)
        _, _, test_loader = cifar100_dataloaders(data_dir='cifar100')
    elif dataset == 'tiny':
        model = resnet50_official(num_classes=200, cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        test_dir = 'tiny-imagenet/validation/'
        item = datasets.ImageFolder(test_dir, transform=transform_chain)
        _, _, test_loader = tiny_imagenet_dataloaders(data_dir='tiny-imagenet')

elif model_type == 'resnet18':
    if dataset == 'cifar10':
        model = resnet18(num_classes=10, cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        item = datasets.CIFAR10(root='cifar10', train=False, transform=transform_chain, download=True)
        _, _, test_loader = cifar10_dataloaders(data_dir='cifar10')
    elif dataset == 'cifar100':
        model = resnet18(num_classes=100, cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        item = datasets.CIFAR100(root='cifar100', train=False, transform=transform_chain, download=True)
        _, _, test_loader = cifar100_dataloaders(data_dir='cifar100')
    elif dataset == 'tiny':
        model = resnet18(num_classes=200, cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        test_dir = 'tiny-imagenet/validation/'
        item = datasets.ImageFolder(test_dir, transform=transform_chain)
        _, _, test_loader = tiny_imagenet_dataloaders(data_dir='tiny-imagenet')

elif model_type == 'vgg16':
    if dataset == 'cifar10':
        model = vgg(16, dataset='cifar10', cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        item = datasets.CIFAR10(root='cifar10', train=False, transform=transform_chain, download=True)
        _, _, test_loader = cifar10_dataloaders(data_dir='cifar10')
    elif dataset == 'cifar100':
        model = vgg(16, dataset='cifar100', cfg=cfg, seed=0)
        model.load_state_dict(fix_robustness_ckpt(torch.load(final_weights)), strict=False)
        item = datasets.CIFAR100(root='cifar100', train=False, transform=transform_chain, download=True)
        _, _, test_loader = cifar100_dataloaders(data_dir='cifar100')

model = model.cuda()
# model = nn.DataParallel(model).cuda()

# =============================================== AA EVAL ==============================================================
## AA EVAL ##

if aa_eval:
    model = model.eval()
    test_loader = data.DataLoader(item, batch_size=128, shuffle=False, num_workers=0)
    from autoattack import AutoAttack
    log = 'store/'+log_folder+'/eval-new2.txt'
    model = model.cuda()
    adversary = AutoAttack(model, norm='Linf', eps=8/255, log_path=log,version='standard')
    adversary.attacks_to_run = ['apgd-t']
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0).cuda()
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0).cuda()
    clean = adversary.clean_accuracy(x_test, y_test,bs=128)
    print('clean',clean)
    adv_complete = adversary.run_standard_evaluation(x_test, y_test,bs=128)
    save_dir = 'store/'+log_folder
    print(adv_complete)
    torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
    save_dir, 'aa', 'standard', adv_complete.shape[0], 8/255))

criterion = nn.CrossEntropyLoss()

# =============================================== PGD20 EVAL ===========================================================
# ## PGD20 EVAL ##

pgd20 = eval_adv(test_loader, model, criterion, 20)
pgd10 = eval_adv(test_loader, model, criterion, 10)

with open('store/'+log_folder+'/eval-new2.txt', 'a') as f:
    f.write(f'\n[PGD-20 VAL] {pgd20}\n')
    f.write(f'[PGD-10 VAL] {pgd10}')
