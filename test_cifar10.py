# %%
# File Name : cat_vs_dog.py
# Purpose :  Training a cat vs dog classifier
# Creation Date : 01-05-2022
# Last Modified : 
# Created By : vamshi
import torch
import torch.nn as nn
from torch import optim
import numpy as np


from data_loader import get_dataloaders
from models import resnet_cifar, resnet9

from pytorch_trainer import Trainer
from pytorch_trainer.metric import Accuracy

from utils import tile_predict, tilify, overlay
# %%
### General hyper-parameters
general_options = {
    'use_cuda' :          True,         # use GPU ?
    'use_tensorboard' :   False,         # Use Tensorboard for saving hparams and metrics ?
    'tensorboard_weight_hist': False    # If save the histogram of model's weight at each epoch
}

### Training hyper-parameters
trainer_args = {
    'epochs' : 200, 
    'loss_fn' : nn.CrossEntropyLoss, ## must be of type nn.Module
    'optimizer' : optim.SGD,
    'loss_fn_kwargs': {},
    'optimizer_kwargs' : {'lr' : 0.01, 'momentum':0.9, 'weight_decay':5e-4},
    'lr_scheduler' : None, 
    'lr_scheduler_kwargs' : {},
    'metric': Accuracy,  ## must be of type metric.Metric or its derived
    'metric_kwargs': {},
    'save_best' : True,
    'save_location' : './saved_models',
    'save_name' : 'inspect',
    'continue_training_saved_model' : None,
}

dataloader_args = {
    'dataset': 'cifar10',
    'batch_size' : 128,
    'shuffle' : False, 
    'num_workers': 12,
    'crop_size': 32
}

network_args = {
    'pretrained': False,
    'num_classes': 10,
    # 'blocks':2,
    # 'freeze_block1':None
}

experiment_summary = 'Training cifar10 for baseline. Network: resnet50'
# %%
trainer = Trainer('inspect', general_options, experiment_summary=experiment_summary)
trainer.initialize_dataloaders(get_dataloaders, **dataloader_args)
# %%
trainer.build_model(resnet_cifar.resnet18, **network_args)
# %%
trainer.model.load_state_dict(torch.load('saved_models/cifar10_cropped_16_resnet18_best.pth')['state_dict'])
trainer.model.to(trainer.device)
trainer.model.eval()
# %%
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from copy import copy
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
classDict = {'0':'plane', '1':'car','2': 'bird', '3':'cat', '4':'deer',
                     '5':'dog', '6':'frog', '7':'horse', '8':'ship', '9':'truck'}
# classDict = {'0':'plane', '1':'bird','2': 'car', '3':'cat', '4':'deer',
#                    '5':'dog', '6':'horse', '7':'monkey', '8':'ship', '9':'truck'}
# %%
test_dataset = datasets.CIFAR10('/home/vamshi/datasets/CIFAR_10_data/', download=False, train=False)
# %%
my_cmap = copy(cm.get_cmap('jet'))
def plot_tensor(x):
    plt.imshow(x.detach().cpu().numpy().transpose(1,2,0),cmap=my_cmap)
# %%
itr = iter(trainer.valloader)
# %%
im,lab = next(itr)
# %%
im,lab = im.to(trainer.device), lab.to(trainer.device)
# %%
idx = 4
plot_tensor(im[idx])
plt.title(classDict[str(lab[idx].item())])
# %%
x = tilify(im[idx].unsqueeze(dim=0), tile_size=16)
# %%
ops = tile_predict(x, trainer.model, batch_size=1024)
# %%
ops = ops.reshape(32,32,10)
# %%
heatmap, predmap = torch.max(ops, dim=2)
heatmap = heatmap.detach().cpu().numpy()
predmap = predmap.detach().cpu().numpy()
# %%
o = overlay(test_dataset.data[idx], (predmap == lab[idx].item())*heatmap)
# %%
plt.imshow(o)
# %%

# %%

# %%
for idx, (im,lab) in enumerate(trainer.valloader):
    im,lab = im.to(trainer.device), lab.to(trainer.device)

    ## compute predictions on each tile
    i = 0

    ## pad the borders of image to get 32*32 number of 8x8 tiles
    ip = torch.nn.functional.pad(im[0], (4,3,4,3))
    predx = []
    heatmapx = []
    while i+7<=ip.shape[-2]-1:
        predy = []
        heatmapy = []
        j = 0
        while j+7<=ip.shape[-1]-1:
            tile = ip[:, i:i+8, j:j+8]
            tile = tile.unsqueeze(dim=0)
            op = trainer.model(tile)
            pred = np.argmax(op.detach().cpu().numpy(), axis=1)
            heatmap = np.max(op.detach().cpu().numpy(), axis=1)
            predy.append(pred.item())
            heatmapy.append(heatmap.item())
            j += 1
        heatmapx.append(heatmapy)
        predx.append(predy)
        i += 1
    predx = np.array(predx)
    heatmapx = np.array(heatmapx)

    overall_pred = trainer.model(im).detach().cpu().numpy().argmax(axis=1)[0].item()
    ## overlay the image onto heatmap
    o1 = overlay(test_dataset.data[idx], (predx == lab[0].item())*heatmapx)
    
    if overall_pred == lab[0].item():
        fig, ax1 = plt.subplots(1,2)
        ax1[0].imshow(test_dataset.data[idx])
        ax1[0].set_title(classDict[str(lab[0].item())], color='orangered')
        ax1[1].imshow(o1)
        ax1[1].set_title(classDict[str(lab[0].item())]+' heatmap', color='orangered')

        plt.savefig('correct/'+str(idx)+'.png', dpi=300)
    else:
        o2 = overlay(test_dataset.data[idx], (predx == overall_pred)*heatmapx)

        fig, ax1 = plt.subplots(1,3)
        ax1[0].imshow(test_dataset.data[idx])
        ax1[0].set_title(classDict[str(lab[0].item())], color='orangered')
        ax1[1].imshow(o1)
        ax1[1].set_title(classDict[str(lab[0].item())]+' heatmap', color='orangered')
        ax1[2].imshow(o2)
        ax1[2].set_title(classDict[str(overall_pred)]+' heatmap', color='orangered')

        plt.savefig('wrong/'+str(idx)+'.png', dpi=300)
    plt.close()

    if idx == 1000:
        break
    
# %%

# %%
mtr = metric.ConfusionMatrix(num_classes=10)
# %%
trainer.model.eval()
mtr.reset()
for im, lab in trainer.valloader:
    im,lab = im.to(trainer.device), lab.to(trainer.device)
    op = trainer.model(im)
    pred = torch.argmax(op,dim=-1)
    mtr.add(pred.detach().cpu().numpy(), lab.detach().cpu().numpy())
# %%
mtr.value()
# %%
itr = iter(trainer.trainloader)
# %%
im,lab = next(itr)
# %%
# %%

# %%
from models import resnet9
import torch
# %%
model = resnet9.Resnet9()
model.to('cuda')
# %%
x = torch.randn((2,3,32,32))
# %%
x = x.to('cuda')
# %%
x1 = model.conv(x)
x1.shape
# %%
x2 = model(x)
x2.shape
# %%

# %%
import torch.nn as nn
# %%
cc = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=1, padding=0, bias=False)
# %%
o = cc(x)
# %%
o.shape
# %%
import torch
from models import resnet9
# %%
a = torch.load('saved_models/cifar10_increm_resnet_1_best.pth')['state_dict']
# %%
m = resnet9.IncrementalResnet9(blocks=2, freeze_block1='saved_models/cifar10_increm_resnet_1_best.pth')
# %%
model_dict = m.state_dict()
# %%
model_dict['block2.0.weight']
# %%
m.block2[0].weight
# %%
a['block1.0.weight']
# %%
a = torch.load('saved_models/cifar10_cropped_32_resnet18_best.pth')['best_val_metric']
# %%
a
# %%
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in a.items() if k in model_dict}
# %%
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# %%
# 3. load the new state dict
m.load_state_dict(pretrained_dict, strict=False)
# %%
from torchvision import models
# %%
resnet18 = models.resnet18()
# %%
resnet18
# %%

# %%

# %%
# %%
image_uri = './cat_dog.jpeg'
# this is the original "cat dog" image used in the Grad-CAM paper
# check the image with Pillow
im = Image.open(image_uri)
im = im.resize((96,96))
# %%
im = transforms.ToTensor()(im).unsqueeze_(0)
# %%
plot_tensor(im[0])
# %%
im = im.to(trainer.device)
# %%
pred = trainer.model(im)
# %%
pred
# %%
## compute predictions on each tile
i = 0

## pad the borders of image to get 32*32 number of 8x8 tiles
ip = torch.nn.functional.pad(dogs.to('cuda')[12], (24,23,24,23))
predx = []
heatmapx = []
while i+47<=ip.shape[-2]-1:
    predy = []
    heatmapy = []
    j = 0
    while j+47<=ip.shape[-1]-1:
        tile = ip[:, i:i+48, j:j+48]
        tile = tile.unsqueeze(dim=0)
        op = trainer.model(tile)
        pred = np.argmax(op.detach().cpu().numpy(), axis=1)
        heatmap = np.max(op.detach().cpu().numpy(), axis=1)
        predy.append(pred.item())
        heatmapy.append(heatmap.item())
        j += 1
    heatmapx.append(heatmapy)
    predx.append(predy)
    i += 1
predx = np.array(predx)
heatmapx = np.array(heatmapx)
# %%
## overlay the image onto heatmap
plot_tensor(dogs[12])
h = plt.imshow((predx == 5)*heatmapx, cmap='jet', alpha=0.3)
# %%

# %%
dogs = []
for im, lab in trainer.trainloader:
    dogs.append(im[torch.where(lab == 5)])
    #break
# %%
dogs = torch.cat(dogs, dim=0)
# %%
plot_tensor(dogs[12])
# %%
