# %%
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import torchvision
import torch
from utils import tile_predict, tilify, overlay
# %%
import matplotlib.pyplot as plt
from matplotlib import cm
from copy import copy
my_cmap = copy(cm.get_cmap('jet'))
def plot_tensor(x):
    plt.imshow(x.detach().cpu().numpy().transpose(1,2,0),cmap=my_cmap)
# %%
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()
model.to('cuda')
# %%
train_path = '/mnt/hanalei-4TBHDD/datasets/imagenet/train'
transform = transforms.Compose(
    [transforms.CenterCrop(256),
     transforms.ToTensor()
    ]
)
imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
data_loader = torch.utils.data.DataLoader(
    imagenet_data,
    batch_size=256,
    shuffle=True,
    num_workers=0
)
# %%
itr = iter(data_loader)
# %%
im, lab = next(itr)
# %%
while(torch.where(lab == 331)[0].shape[0] == 0):
    im, lab = next(itr)
# %%
idx = 112
plot_tensor(im[idx])
# %%
preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# %%
input_tensor = preprocess(im[idx])
input_batch = input_tensor.unsqueeze(0)
# %%
input_batch = input_batch.to('cuda')
# %%
with torch.no_grad():
    output = model(input_batch)
# %%
torch.topk(output, 5)
# %%
x = tilify(input_batch, tile_size=64)
# %%
ops = tile_predict(x, model, batch_size=1024)
# %%
ops = ops.reshape(256,256,1000)
# %%
heatmap, predmap = torch.max(ops, dim=2)
heatmap = heatmap.detach().cpu().numpy()
predmap = predmap.detach().cpu().numpy()
# %%
dog = overlay(im[idx].detach().cpu().numpy().transpose(1,2,0), (predmap == 355)*heatmap)
# %%
plt.figure(figsize=(10,10))
plt.imshow(dog)
plt.savefig('llama_hm.png', dpi=300)
# %%
plt.figure(figsize=(10,10))
plot_tensor(im[idx][:,40:104,120:184])
plt.savefig('hare_tile.png', dpi=300)
# %%
