import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

torch.manual_seed(42)

cifar10_mean = (0.4914, 0.4822, 0.4465)   ## for CIFAR-10
cifar10_std = (0.247, 0.243, 0.261)

stl10_full_mean = (0.44087802, 0.42790631, 0.38678794) ## train+unlabelled
stl10_full_std = (0.26826769, 0.26104504, 0.26866837)

stl10_train_mean = (0.44671062, 0.43980984, 0.40664645) ## train only
stl10_train_std = (0.26034098, 0.25657727, 0.27126738)

tiny_imagenet_mean = (0.04112063, 0.04112063, 0.04112063) ## train
tiny_imagenet_std = (0.20317943, 0.20317943, 0.20317943)

def get_dataloaders(dataset='cifar10', train_val_split=None, batch_size=32, crop_size=32, train_transform=True, shuffle=True, num_workers=8, **dataset_kwargs):
    """
    Get pytorch dataloaders
    """
    sampler = None
    if dataset == 'cifar10':

        ## Data transforms
        def zero_norm():
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std)
        ])

        # Transforms object for trainset with augmentation
        def crop_hflip():
            return transforms.Compose([
                    #transforms.ToPILImage(),
                    transforms.RandomCrop(crop_size, padding=0),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std)])
        if train_transform:
            trainset = datasets.CIFAR10('/home/vamshi/datasets/CIFAR_10_data/', download=False, train=True, transform=crop_hflip(), **dataset_kwargs)
        else:
            trainset = datasets.CIFAR10('/home/vamshi/datasets/CIFAR_10_data/', download=False, train=True, transform=zero_norm(), **dataset_kwargs)
        test_dataset = datasets.CIFAR10('/home/vamshi/datasets/CIFAR_10_data/', download=False, train=False, transform=zero_norm(), **dataset_kwargs)
    elif dataset == 'stl10':
        ## Data transforms
        def zero_norm():
            return transforms.Compose([
                #transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(stl10_train_mean, stl10_train_std)
        ])

        # Transforms object for trainset with augmentation
        def crop_hflip():
            return transforms.Compose([
                    #transforms.ToPILImage(),
                    #transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
                    transforms.RandomCrop(crop_size, padding=0),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(),
                    transforms.Normalize(stl10_train_mean, stl10_train_std)
                ])
        if train_transform:
            trainset = datasets.STL10('/home/vamshi/datasets/STL10/', download=False, split='train', transform=crop_hflip(), **dataset_kwargs)
        else:
            trainset = datasets.STL10('/home/vamshi/datasets/STL10/', download=False, split='train', transform=zero_norm(), **dataset_kwargs)
        test_dataset = datasets.STL10('/home/vamshi/datasets/STL10/', download=False, split='test', transform=zero_norm(), **dataset_kwargs)
    elif dataset == 'cifar100':
        def zero_norm():
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std)
        ])

        # Transforms object for trainset with augmentation
        def crop_hflip():
            return transforms.Compose([
                    #transforms.ToPILImage(),
                    transforms.RandomCrop(crop_size, padding=0),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std)])

        trainset = datasets.CIFAR100('/home/vamshi/datasets/CIFAR_100_data/', download=False, train=True, transform=crop_hflip(), **dataset_kwargs)
        test_dataset = datasets.CIFAR100('/home/vamshi/datasets/CIFAR_100_data/', download=False, train=False, transform=zero_norm(), **dataset_kwargs)
    elif dataset == 'tiny_imagenet':
        def zero_norm():
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(tiny_imagenet_mean, tiny_imagenet_std)
        ])

        # Transforms object for trainset with augmentation
        def crop_hflip():
            return transforms.Compose([
                    #transforms.ToPILImage(),
                    transforms.RandomCrop(crop_size, padding=0),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(),
                    transforms.Normalize(tiny_imagenet_mean, tiny_imagenet_std)
                ])
                    
        trainset = datasets.ImageFolder(root='/home/vamshi/datasets/tiny-imagenet-200/train/', transform=crop_hflip())
        test_dataset = datasets.ImageFolder(root='/home/vamshi/datasets/tiny-imagenet-200/val/', transform=zero_norm())
    else:
        print("Dataset: {} not supported!".format(dataset))
        return
    
    if train_val_split is not None:
        # train and val
        train_size = int(train_val_split * len(trainset))
        val_size = len(trainset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
        
        # test    
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler)
        valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return trainloader,valloader,testloader

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler = sampler)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return trainloader,testloader