# File Name : cifar100_resnet.py
# Purpose :  Cifar100 training on tiles
# Creation Date : 01-05-2022
# Last Modified : 
# Created By : 

import torch.nn as nn
from torch import optim
import numpy as np
import torch

from data_loader import get_dataloaders
from models import resnet9, resnet_cifar

from pytorch_trainer import Trainer
from pytorch_trainer.metric import Accuracy

### General hyper-parameters
general_options = {
    'use_cuda' :          True,         # use GPU ?
    'use_tensorboard' :   True,         # Use Tensorboard for saving hparams and metrics ?
    'tensorboard_weight_hist': False    # If save the histogram of model's weight at each epoch
}

### Training hyper-parameters
trainer_args = {
    'epochs' : 400,
    'loss_fn' : nn.CrossEntropyLoss, ## must be of type nn.Module
    'optimizer' : optim.Adam,
    'loss_fn_kwargs': {},
    'optimizer_kwargs' : {'lr' : 0.002, 'weight_decay':1e-4},
    'lr_scheduler' : optim.lr_scheduler.MultiStepLR, 
    'lr_scheduler_kwargs' : {'milestones':[40, 80, 100, 150, 200, 250, 300], 'gamma':0.5},
    'metric': Accuracy,  ## must be of type metric.Metric or its derived
    'metric_kwargs': {},
    'save_best' : True,
    'save_location' : './saved_models',
    'save_name' : 'cifar10_cropped_30_resnet18',
    'continue_training_saved_model' : None,
}

dataloader_args = {
    'dataset': 'cifar10',
    'batch_size' : 128,
    'shuffle' : True, 
    'num_workers': 12,
    'crop_size': 30
}

network_args = {
    'pretrained': False,
    'num_classes': 10,
    #'blocks':2,
    #'freeze_block1':None
}

experiment_summary = 'Training resnet18 on 30x30 cifar10 images'

if __name__ == '__main__':
    trainer = Trainer('cifar10_cropped_30_resnet18', general_options, experiment_summary=experiment_summary)
    trainer.initialize_dataloaders(get_dataloaders, **dataloader_args)
    trainer.build_model(resnet_cifar.resnet18, **network_args)
    #trainer.model.load_state_dict(torch.load('saved_models/cifar10_cropped_resnet_16_best.pth')['state_dict'])
    # trainer.model.to(trainer.device)
    trainer.train(**trainer_args)
