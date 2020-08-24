import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_color_mask(object_index, nc=None):
    """ Colors each index differently. Useful for visualizing semantic masks

        @param object_index: a [H x W] numpy array of ints from {0, ..., nc-1}
        @param nc: total number of colors. If None, this will be inferred by masks

        @return: a [H x W x 3] numpy array of dtype np.uint8
    """
    object_index = object_index.astype(int)

    if nc is None:
        NUM_COLORS = object_index.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    color_mask = np.zeros(object_index.shape + (3,)).astype(np.uint8)
    for i in np.unique(object_index):
        if i == 0 or i == -1:
            continue
        color_mask[object_index == i, :] = np.array(colors[i][:3]) * 255
        
    return color_mask


def build_matrix_of_indices(height, width):
    """ Builds a [height, width, 2] numpy array containing coordinates.

        @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1,2,0)


def concatenate_spatial_coordinates(feature_map):
    """ Adds x,y coordinates as channels to feature map

        @param feature_map: a [T x C x H x W] torch tensor
    """
    T, C, H, W = feature_map.shape

    # build matrix of indices. then replicated it T times
    MoI = build_matrix_of_indices(H, W) # Shape: [H, W, 2]
    MoI = np.tile(MoI, (T, 1, 1, 1)) # Shape: [T, H, W, 2]
    MoI[..., 0] = MoI[..., 0] / (H-1) * 2 - 1 # in [-1, 1]
    MoI[..., 1] = MoI[..., 1] / (W-1) * 2 - 1
    MoI = torch.from_numpy(MoI).permute(0,3,1,2).to(feature_map.device) # Shape: [T, 2, H, W]

    # Concatenate on the channels dimension
    feature_map = torch.cat([feature_map, MoI], dim=1)

    return feature_map


def visualize_segmentation(im, masks, nc=None):
    """ Visualize segmentations nicely. Based on code from:
        https://github.com/roytseng-tw/Detectron.pytorch/blob/master/lib/utils/vis.py

        @param im: a [H x W x 3] RGB image. numpy array of dtype np.uint8
        @param masks: a [H x W] numpy array of dtype np.uint8 with values in {0, ..., nc-1}
        @param nc: total number of colors. If None, this will be inferred by masks

        @return: a [H x W x 3] numpy array of dtype np.uint8
    """ 
    from matplotlib.patches import Polygon

    masks = masks.astype(int)
    im = im.copy()

    # Generate color mask
    if nc is None:
        NUM_COLORS = masks.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    # Mask
    imgMask = np.zeros(im.shape)


    # Draw color masks
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Add to the mask
        imgMask[e] = color_mask

    # Add the mask to the image
    imgMask = (imgMask * 255).round().astype(np.uint8)
    im = cv2.addWeighted(im, 0.5, imgMask, 0.5, 0.0)


    # Draw mask contours
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Find contours
        contour, hier = cv2.findContours(
            e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # Plot the nice outline
        for c in contour:
            cv2.drawContours(im, contour, -1, (255,255,255), 2)

    return im
    

### The two functions below were adatped from the DAVIS public dataset ###

def imread_indexed(filename):
    """ Load segmentation image (with palette) given filename."""
    im = Image.open(filename)
    annotation = np.array(im)
    return annotation


def imwrite_indexed(filename,array):
    """ Save indexed png with palette."""

    palette_abspath = '/home/chrisxie/projects/random_stuff/palette.txt' # hard-coded filepath
    color_palette = np.loadtxt(palette_abspath, dtype=np.uint8).reshape(-1,3)

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


def mask_to_tight_box_numpy(mask):
    """ Return bbox given mask

        @param mask: a [H x W] numpy array
    """
    a = np.transpose(np.nonzero(mask))
    bbox = np.min(a[:, 1]), np.min(a[:, 0]), np.max(a[:, 1]), np.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max


def mask_to_tight_box_pytorch(mask):
    """ Return bbox given mask

        @param mask: a [H x W] torch tensor
    """
    a = torch.nonzero(mask)
    bbox = torch.min(a[:, 1]), torch.min(a[:, 0]), torch.max(a[:, 1]), torch.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max


def mask_to_tight_box(mask):
    if type(mask) == torch.Tensor:
        return mask_to_tight_box_pytorch(mask)
    elif type(mask) == np.ndarray:
        return mask_to_tight_box_numpy(mask)
    else:
        raise Exception(f"Data type {type(mask)} not understood for mask_to_tight_box...")


def compute_xyz(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters.
        Assumes camera uses left-handed coordinate system, with 
            x-axis pointing right
            y-axis pointing up
            z-axis pointing "forward"

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used 

        @return: a [H x W x 3] numpy array
    """

    # Compute focal length from camera parameters
    if 'fx' in camera_params and 'fy' in camera_params:
        fx = camera_params['fx']
        fy = camera_params['fy']
    else: # simulated data
        aspect_ratio = camera_params['img_width'] / camera_params['img_height']
        e = 1 / (np.tan(np.radians(camera_params['fov']/2.)))
        t = camera_params['near'] / e; b = -t
        r = t * aspect_ratio; l = -r
        alpha = camera_params['img_width'] / (r-l) # pixels per meter
        focal_length = camera_params['near'] * alpha # focal length of virtual camera (frustum camera)
        fx = focal_length; fy = focal_length

    if 'x_offset' in camera_params and 'y_offset' in camera_params:
        x_offset = camera_params['x_offset']
        y_offset = camera_params['y_offset']
    else: # simulated data
        x_offset = camera_params['img_width']/2
        y_offset = camera_params['img_height']/2

    indices = build_matrix_of_indices(camera_params['img_height'], camera_params['img_width'])
    indices[..., 0] = np.flipud(indices[..., 0]) # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]

    return xyz_img


def seg2bmap(seg, return_contour=False):
    """ From a segmentation, compute a binary boundary map with 1 pixel wide
        boundaries. This boundary lives on the mask, i.e. it's a subset of the mask.

        @param seg: a [H x W] numpy array of values in {0,1}

        @return: a [H x W] numpy array of values in {0,1}
                 a [2 x num_boundary_pixels] numpy array. [0,:] is y-indices, [1,:] is x-indices
    """
    seg = seg.astype(np.uint8)
    contours, hierarchy = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    temp = np.zeros_like(seg)
    bmap = cv2.drawContours(temp, contours, -1, 1, 1)

    if return_contour: # Return the SINGLE largest contour
        contour_sizes = [len(c) for c in contours]
        ind = np.argmax(contour_sizes)
        contour = np.ascontiguousarray(np.fliplr(contours[ind][:,0,:]).T) # Shape: [2 x num_boundary_pixels]
        return bmap, contour
    else:
        return bmap
    

def largest_connected_component(mask, connectivity=4):
    """ Run connected components algorithm and return mask of largest one

        @param mask: a [H x W] numpy array 

        @return: a [H x W] numpy array of same type as input
    """

    # Run connected components algorithm
    num_components, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)

    # Find largest connected component via set distance
    largest_component_num = -1
    largest_component_size = -1 
    for j in range(1, num_components):
        component_size = np.count_nonzero(components == j)
        if component_size > largest_component_size:
            largest_component_num = j
            largest_component_size = component_size

    return (components == largest_component_num).astype(mask.dtype)


def torch_to_numpy(torch_tensor, is_standardized_image = False):
    """ Converts torch tensor (NCHW) to numpy tensor (NHWC) for plotting
    
        If it's an rgb image, it puts it back in [0,255] range (and undoes ImageNet standardization)
    """
    np_tensor = torch_tensor.cpu().clone().detach().numpy()
    if np_tensor.ndim == 4: # NCHW
        np_tensor = np_tensor.transpose(0,2,3,1)
    if is_standardized_image:
        _mean=[0.485, 0.456, 0.406]; _std=[0.229, 0.224, 0.225]
        for i in range(3):
            np_tensor[...,i] *= _std[i]
            np_tensor[...,i] += _mean[i]
        np_tensor *= 255
            
    return np_tensor


def subplotter(images, titles=None, fig_num=1, plot_width=5):
    """ Function for plotting side-by-side images."""

    num_images = len(images)
    fig = plt.figure(fig_num, figsize=(num_images*plot_width, plot_width))

    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        if titles:
            plt.title(titles[i])

  