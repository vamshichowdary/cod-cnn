# %%
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
# %%
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()
# %%
image_uri = './cat_dog.jpeg'
# this is the original "cat dog" image used in the Grad-CAM paper
# check the image with Pillow
im = Image.open(image_uri)
im = im.crop((100,0,580,480))
input_image = im.resize((256,256))
# %%
preprocess = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
# %%
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
# %%
with torch.no_grad():
    output = model(input_batch[:,:,:64,50:114])
# %%
torch.topk(output, 5)
# %%
import matplotlib.pyplot as plt
from matplotlib import cm
from copy import copy
my_cmap = copy(cm.get_cmap('jet'))
def plot_tensor(x):
    plt.imshow(x.detach().cpu().numpy().transpose(1,2,0),cmap=my_cmap)
# %%
plot_tensor(input_batch[0, :, 50:114,50:114])
# %%
i = 0
ip = torch.nn.functional.pad(input_batch[0], (32,31,32,31))
predx = []
heatmapx = []
while i+63<=ip.shape[-2]-1:
    predy = []
    heatmapy = []
    j = 0
    while j+63<=ip.shape[-1]-1:
        tile = ip[:, i:i+64, j:j+64]
        tile = tile.unsqueeze(dim=0)
        with torch.no_grad():
            op = model(tile)
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
plot_tensor(input_batch[0])
h = plt.imshow((predx == 242)*heatmapx, cmap='jet', alpha=0.3)
# %%
x = torch.nn.functional.pad(input_batch, (32,31,32,31))
# %%
a = x.unfold(3, 64, 1).unfold(2, 64, 1)
# %%
a.shape
# %%
def tilify(x, tile_size):
    """
    Tiles the input tensor x. If x has HxW spatial size, zero padding is done before tiling so that
    H*W number of tiles are obtained. Stride = 1 only supported for now
    Args:
        x (Tensor): input tensor of shape BxCxHxW, where B is the number of batches
        tile_size (int): size of the tiles
    """
    assert x.ndim == 4
    B, C, W, H = x.shape
    ## zero padding 
    if tile_size % 2 == 0:
        pad_size = tile_size // 2
        x = torch.nn.functional.pad(x, (pad_size,pad_size-1,pad_size,pad_size-1))
    else:
        pad_size = (tile_size-1) // 2
        x = torch.nn.functional.pad(x, (pad_size,pad_size,pad_size,pad_size))

    stride = 1
    ## do the tiling
    tiles = x.unfold(3, tile_size, stride).unfold(2, tile_size, stride)
    tiles = tiles.contiguous().view(B, C, -1, tile_size, tile_size)
    tiles = tiles.permute(0, 2, 1, 3, 4)
    tiles = tiles.view(-1 , C, tile_size, tile_size)
    return tiles
# %%
def tile_predict(tiles, model, batch_size):
    """
    Get predictions on tiles in batches, since inputting all tiles into model gives OOM
    """
    assert tiles.ndim == 4
    b = batch_size
    ops = []
    while b <= tiles.shape[0]:
        with torch.no_grad():
            op = model( tiles[b - batch_size : min(b, tiles.shape[0])] )
        ops.append(op)
        b += batch_size
    ops = torch.cat(ops, dim=0)
    return ops
# %%
import cv2
# %%
def overlay(base, heatmap):
    """
    Overlays a 2-channel "heatmap" onto a base image
    """
    img = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img = cv2.resize(img, (256, 256))

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.resize(heatmap, (256, 256))

    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

    heatmap_img[:,:,0] = np.where(heatmap>0, heatmap_img[:,:,0], 0)
    heatmap_img[:,:,1] = np.where(heatmap>0, heatmap_img[:,:,1], 0)
    heatmap_img[:,:,2] = np.where(heatmap>0, heatmap_img[:,:,2], 0)

    fin = cv2.addWeighted(heatmap_img, 0.5, img, 1.0, 0)
    return fin
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
cat = overlay(np.asarray(im.resize((256,256))), (predmap == 282)*heatmap)
dog = overlay(np.asarray(im.resize((256,256))), (predmap == 242)*heatmap)
# %%
fig, ax1 = plt.subplots(1,2)
ax1[0].set_axis_off()
ax1[0].imshow(cat)
ax1[0].set_title('tiger-cat heatmap', color='orangered')
ax1[1].set_axis_off()
ax1[1].imshow(dog)
ax1[1].set_title('boxer heatmap', color='orangered')
# %%
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(np.asarray(im.resize((256,256)))[180:244,100:164,:])
plt.savefig('cat', dpi=300)
# %%
#plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(np.asarray(im.resize((256,256)))[20:84,100:164,:])
plt.savefig('dog', dpi=300)
# %%
