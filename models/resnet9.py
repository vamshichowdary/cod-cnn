import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out += residual
        return out


class Resnet9(nn.Module):
    """
    A Residual network.
    """
    def __init__(self):
        super(Resnet9, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        self.fc = nn.Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out

class IncrementalResnet9(nn.Module):
    """
    Rewriting Resnet9 in terms of two blocks each with one Residual Block layer.
    Just for ease of copying and freezing weights of individual blocks from trained models.
    """
    def __init__(self, blocks, freeze_block1 = None, num_classes=10):
        super(IncrementalResnet9, self).__init__()

        self.blocks = blocks

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        )

        if freeze_block1 is not None:
            self.load_and_freeze_module(freeze_block1, self.block1, 'block1')

        if blocks != 1:
            self.block2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=256, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=256, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            )

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        if blocks == 1:
            conv_out_features = 512
        else:
            conv_out_features = 1024
        self.fc = nn.Linear(in_features=conv_out_features, out_features=num_classes, bias=True)
    
    def forward(self, x):
        out = self.block1(x)
        if self.blocks != 1:
            out = self.block2(out)
        out = self.avgpool(out)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out

    def load_and_freeze_module(self,load_path,mod,mod_name):
        """
        Loads a pretrained module from `load_path`, copies its weights into `mod`.
        Copies only the params which match the same name as in the `load_path` module\'s
        Freezes all the params in the `mod`.
        """
        ## get the state dict of module
        mod_dict = mod.state_dict()
        ## load the pretrained weights of block1
        pretrained = torch.load(load_path)['state_dict']
        print('Loaded state dict from {} and copying to {}'.format(load_path, mod_name))
        # # 1. filter out unnecessary keys
        mod_dict_keys_map = {mod_name+'.'+k : k for k in mod_dict.keys()}
        pretrained_dict = {mod_dict_keys_map[k]: v for k, v in pretrained.items() if k in mod_dict_keys_map}
        # # 2. load the new state dict
        mod.load_state_dict(pretrained_dict)
        # # 3. Freeze the pretrained weights
        for param in mod.parameters():
            param.requires_grad = False
        print('Frozen weights of {}!!'.format(mod_name))

class FrozenResnet9(Resnet9):
    """
    Resnet9 with conv layer weights frozen. Final FC layer weights are NOT frozen
    """
    def __init__(self, num_fc_features = None):
        super(FrozenResnet9, self).__init__()
        for param in self.conv.parameters():
            param.requires_grad = False
        
        if num_fc_features is not None:
            self.fc = nn.Sequential(
                nn.Linear(in_features=512, out_features=num_fc_features, bias=True),
                nn.Linear(in_features=num_fc_features, out_features=10, bias=True)
            )