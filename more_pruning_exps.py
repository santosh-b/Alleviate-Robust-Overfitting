import sys
from utils import *
sys.path.append('Synaptic-Flow')
from Models.imagenet_resnet import resnet18 as pruned_resnet18
from Utils.generator import *
from utils import *
from Pruners.pruners import *
import matplotlib.pyplot as plt
import torch.nn.utils.prune as pruning

import argparse
import os

## helpers ##

def prune_scoremask_model(model, pct):
    for m0 in model.modules():
        if isinstance(m0,nn.Conv2d):
            pruning.ln_structured(
                m0, 'weight', amount=pct, dim=1, n=1
            )
    return modelnew

def rewind_score_mask_model(model, pct):
    modelnew = pruned_resnet18(input_shape=(32,32), num_classes=10).cuda()
    for [m0,m1] in zip(model.modules(), modelnew.modules()):
        if (isinstance(m0,nn.Conv2d) and isinstance(m1, nn.Conv2d)):
            pruning.ln_structured(
                m0, 'weight', amount=pct, dim=1, n=1
            )
            pruning.CustomFromMask.apply(m1, 'weight', m0.weight_mask)
    return modelnew

def get_score_mask_as_model(pruner, original_model):
    new_model = pruned_resnet18(input_shape=(32,32), num_classes=10)
    mask = {}
    for name, m in original_model.named_parameters():
        try:
            data = pruner.scores[id(m)]
            mask[name] = data
        except:
            continue        
    for name, m in new_model.named_parameters():
        if name in mask:
            m.data = mask[name]
    return new_model

def log(model, val_sa, val_ra, test_sa, test_ra, epoch, args):
    log_folder = args.save_dir
    with open(str(args.save_dir)+'/log.txt', 'a') as f:
        f.write(str(epoch)+' '+
                str(test_sa)+' '+
                str(test_ra)+' '+
                str(val_sa)+' '+
                str(val_ra)+'\n')
    if epoch == 109:
        torch.save(model.state_dict(), log_folder+f'/{epoch}_checkpoint.pt')

###

parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')

parser.add_argument('--pruning_mode', type=str, required=True)
parser.add_argument('--pct', type=float, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--print_freq', default=50, type=int, help='logging frequency during training')

########################## attack setting ##########################
parser.add_argument('--norm', default='linf', type=str, help='linf or l2')
parser.add_argument('--train_eps', default=8, type=float, help='epsilon of attack during training')
parser.add_argument('--train_step', default=10, type=int, help='itertion number of attack during training')
parser.add_argument('--train_gamma', default=2, type=float, help='step size of attack during training')
parser.add_argument('--train_randinit', action='store_false', help='randinit usage flag (default: on)')
parser.add_argument('--test_eps', default=8, type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2, type=float, help='step size of attack during testing')
parser.add_argument('--test_randinit', action='store_false', help='randinit usage flag (default: on)')

args = parser.parse_args()

args.save_dir = 'store/'+str(args.save_dir)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

mode = args.pruning_mode
pct = args.pct

args.train_eps = args.train_eps / 255
args.train_gamma = args.train_gamma / 255
args.test_eps = args.test_eps / 255
args.test_gamma = args.test_gamma / 255

model = pruned_resnet18(input_shape=(32,32), num_classes=10).cuda()
train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = 64, data_dir='cifar10')

decreasing_lr = list(map(int, '100,105'.split(',')))
criterion = nn.CrossEntropyLoss()
lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=.9,
                            weight_decay=5e-4)

train_acc = train_it(train_loader, model, criterion, optimizer, 0, None)

if mode == 'snip':
    pruner = SNIP(masked_parameters(model))
elif mode == 'grasp':
    pruner = GraSP(masked_parameters(model))
elif mode == 'random':
    pass

pruner.score(model, nn.CrossEntropyLoss(), train_loader, torch.device('cuda'))

model_score = get_score_mask_as_model(pruner, model)
new_model = rewind_score_mask_model(model_score, pct)



decreasing_lr = list(map(int, '100,105'.split(',')))
criterion = nn.CrossEntropyLoss()
lr = 0.1
optimizer = torch.optim.SGD(new_model.parameters(), lr,
                            momentum=.9,
                            weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

all_result = {}
all_result['train_acc'] = []
all_result['val_sa'] = []
all_result['val_ra'] = []
all_result['test_sa'] = []
all_result['test_ra'] = []
best_sa = 0
best_ra = 0

for epoch in range(110):
    print(new_model.conv1.weight[0])
    print(optimizer.state_dict()['param_groups'][0]['lr'])

    print('baseline adversarial training')
    train_acc = train_epoch_adv(train_loader, new_model, criterion, optimizer, epoch, args)

    all_result['train_acc'].append(train_acc)
    scheduler.step()

    ###validation###
    val_sa = test(val_loader, new_model, criterion, args)
    test_sa = test(test_loader, new_model, criterion, args)
    val_ra = test_adv(val_loader, new_model, criterion, args)   
    test_ra = test_adv(test_loader, new_model, criterion, args)  

    all_result['val_sa'].append(val_sa)
    all_result['val_ra'].append(val_ra)
    all_result['test_sa'].append(test_sa)
    all_result['test_ra'].append(test_ra)

    is_sa_best = val_sa  > best_sa
    best_sa = max(val_sa, best_sa)

    is_ra_best = val_ra  > best_ra
    best_ra = max(val_ra, best_ra)

    checkpoint_state = {
        'best_sa': best_sa,
        'best_ra': best_ra,
        'epoch': epoch+1,
        'state_dict': new_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'result': all_result
    }

    # if args.swa and epoch >= args.swa_start and (epoch - args.swa_start) % args.swa_c_epochs == 0:

    #     # SWA
    #     moving_average(swa_model, model, 1.0 / (swa_n + 1))
    #     swa_n += 1
    #     bn_update(train_loader, swa_model)

    #     val_sa_swa = test(val_loader, swa_model, criterion, args)
    #     val_ra_swa = test_adv(val_loader, swa_model, criterion, args)   
    #     test_sa_swa = test(test_loader, swa_model, criterion, args)
    #     test_ra_swa = test_adv(test_loader, swa_model, criterion, args)  

    #     all_result['val_sa_swa'].append(val_sa_swa)
    #     all_result['val_ra_swa'].append(val_ra_swa)
    #     all_result['test_sa_swa'].append(test_sa_swa)
    #     all_result['test_ra_swa'].append(test_ra_swa)

    #     is_sa_best_swa = val_sa_swa  > best_sa_swa
    #     best_sa_swa = max(val_sa_swa, best_sa_swa)

    #     is_ra_best_swa = val_ra_swa  > best_ra_swa
    #     best_ra_swa = max(val_ra_swa, best_ra_swa)

    #     checkpoint_state.update({
    #         'swa_state_dict': swa_model.state_dict(),
    #         'swa_n': swa_n,
    #         'best_sa_swa': best_sa_swa,
    #         'best_ra_swa': best_ra_swa
    #     })

    # elif args.swa:

    #     all_result['val_sa_swa'].append(val_sa)
    #     all_result['val_ra_swa'].append(val_ra)
    #     all_result['test_sa_swa'].append(test_sa)
    #     all_result['test_ra_swa'].append(test_ra)

    checkpoint_state.update({
        'result': all_result
    })
    save_checkpoint(checkpoint_state, is_sa_best, is_ra_best, False, False, args.save_dir)

    log(new_model, val_sa, val_ra, test_sa, test_ra, epoch, args)

    plt.plot(all_result['train_acc'], label='train_acc')
    plt.plot(all_result['test_sa'], label='SA')
    plt.plot(all_result['test_ra'], label='RA')

    # if args.swa:
    #     plt.plot(all_result['test_sa_swa'], label='SWA_SA')
    #     plt.plot(all_result['test_ra_swa'], label='SWA_RA')

    plt.legend()
    plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
    plt.close()

