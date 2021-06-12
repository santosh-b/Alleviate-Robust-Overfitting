import models
import torch


model = models.__dict__['resnet50_prune'](pretrained=False)
print(model)

checkpoint = torch.load('/Users/mahaoyu/Downloads/pruned_5008_0.5/pruned.pth.tar', map_location='cpu')
cfg_input = checkpoint['cfg']
model = models.__dict__['resnet50_prune'](pretrained=False, cfg=cfg_input)
print(model)


state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)


