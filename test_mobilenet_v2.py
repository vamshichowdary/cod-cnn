# %%
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from utils import tile_predict, tilify, overlay
# %%
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()
model.to('cuda')
# %%
image_uri = './cat_dog.jpeg'
# this is the original "cat dog" image used in the Grad-CAM paper
# check the image with Pillow
im = Image.open(image_uri)
im = im.crop((100,0,580,480))
input_image = im.resize((256,256))
#input_image = im
# %%
# image_uri = 'n02108000_EntleBucher.jpeg'
# im = Image.open(image_uri)
# im = im.crop((60,0,432,372))
# input_image = im.resize((256,256))
# %%
preprocess = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#%%
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
# %%
input_batch = input_batch.to('cuda')
# %%
with torch.no_grad():
    output = model(input_batch)
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
dog = overlay(np.asarray(im.resize((256,256))), (predmap == 242)*heatmap)
# %%
plt.imshow(dog)
#plt.imshow((predmap == 242)*heatmap)
# %%
plt.axis('off')
plt.imshow(np.asarray(im)[50:114,200:264,:])
# %%
cat = overlay(np.asarray(im.resize((256,256))), (predmap == 282)*heatmap)
dog = overlay(np.asarray(im.resize((256,256))), (predmap == 242)*heatmap)
# %%
fig, ax1 = plt.subplots(1,2,figsize=(10,7))
ax1[0].set_axis_off()
ax1[0].imshow(cat)
ax1[0].set_title('tiger-cat heatmap', color='orangered')
ax1[1].set_axis_off()
ax1[1].imshow(dog)
ax1[1].set_title('boxer heatmap', color='orangered')
plt.savefig('cat_dog_hm.png', dpi=300)
# %%
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(np.asarray(im.resize((256,256)))[180:244,100:164,:])
plt.savefig('cat', dpi=300)
# %%
#plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(np.asarray(im.resize((256,256)))[50:114,50:114,:])
#plt.savefig('dog', dpi=300)
# %%
