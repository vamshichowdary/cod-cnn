import cv2
import torch
import numpy as np

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
            #op = torch.nn.functional.softmax(op, dim=1)
        ops.append(op.detach().cpu())
        b += batch_size
    ops = torch.cat(ops, dim=0)
    return ops

def overlay(base, heatmap):
    """
    Overlays a 2-channel "heatmap" onto a base image
    """
    size = base.shape[0]
    img = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img = cv2.resize(img, (size,size))

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.resize(heatmap, (size,size))

    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

    heatmap_img[:,:,0] = np.where(heatmap>0, heatmap_img[:,:,0], 0)
    heatmap_img[:,:,1] = np.where(heatmap>0, heatmap_img[:,:,1], 0)
    heatmap_img[:,:,2] = np.where(heatmap>0, heatmap_img[:,:,2], 0)

    fin = cv2.addWeighted(heatmap_img, 0.5, img, 1.0, 0)
    return fin

def generate_sample_heatmap(image_location, tile_size, heatmap_class=None):
    """
    Generates a sample heat map using tile size as tile_size using mobilenet_v2 downloaded from pytorch's model zoo
    Args:
        image_location: path of image
        tile_size: tile size using which heat maps is to be generated
        heatmap_class: the class for which heatmap needs to be generated. If None, will generate the heatmap for the predicted class
    """
    from PIL import Image
    from torchvision import transforms
    import torch
    import matplotlib as plt

    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    ## download the pretrained model from pytorch model zoo
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    ## load the image
    im = Image.open(image_location)

    ## preprocess with imagenet preprocessing pipeline
    input_image = im.resize((256,256))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model  

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    ## pass the image through model    
    with torch.no_grad():
        output = model(input_batch)
    predicted_class = output.argmax(dim=1)[0].item()
    
    ## generate the heatmap
    x = tilify(input_batch, tile_size=tile_size)
    ops = tile_predict(x, model, batch_size=1024)
    ops = ops.reshape(256,256,1000)
    heatmap, predmap = torch.max(ops, dim=2)
    heatmap = heatmap.detach().cpu().numpy()
    predmap = predmap.detach().cpu().numpy()

    overlayed_heatmap = overlay(np.asarray(im.resize((256,256))), (predmap == predicted_class)*heatmap)
    plt.imshow(overlayed_heatmap)