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
        ops.append(op)
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