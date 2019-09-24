def visualize_data():
	pass
"""
dl = data_loader.get_TOD_train_dataloader(TOD_filepath, data_loading_params, num_workers=4, shuffle=True)
dl_iter = dl.__iter__()
batch = next(dl_iter)
rgb_imgs = torch_to_numpy(batch['rgb']) # Shape: [N x H x W x 3]
xyz_imgs = torch_to_numpy(batch['xyz']) # Shape: [N x H x W x 3]
labels = torch_to_numpy(batch['labels']) # Shape: [N x H x W]
N, H, W = labels.shape[:3]

--------------------------------------------------------------------------------------------------------------

%matplotlib inline
# Plot

index_to_plot = 4
print(f"Scene dir: {batch['scene_dir'][index_to_plot]}")

plt.figure(1, figsize=(20,60))

# RGB image
rgb_img = rgb_imgs[index_to_plot, ...].astype(np.uint8)
plt.subplot(1,3,1)
plt.imshow(rgb_img)
plt.title('RGB')

# Depth image
depth_img = xyz_imgs[index_to_plot, ..., 2]
plt.subplot(1,3,2)
plt.imshow(depth_img, cmap='gray')
plt.title('Linear depth')

# Segmentation image
seg_img = labels[index_to_plot, ...]
print(np.unique(seg_img))
plt.subplot(1,3,3)
plt.imshow(seg_img)
plt.title('Segmentation')

# plt.savefig(f'/home/chrisxie/Desktop/ex{i}.png', bbox_inches='tight')
# i += 1

"""

def plot_visualizations_from_return_annotations():
	pass
"""
IN data_loader.py

def get_TOD_test_dataloader(base_dir, params, batch_size=8, num_workers=4, shuffle=False):

    dataset = Tabletop_Object_Dataset(base_dir + 'test_set_small/', params)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)


IN SSC_sandbox.iypnb:

dl = data_loader.get_TOD_test_dataloader(TOD_filepath, data_loading_params, batch_size=8, num_workers=8, shuffle=True)
test_results = seg_network.test(dl, foreground_only=True, return_annotations=True)

print('Foreground/Background metrics:')
print(f"\tMean Background IoU: {test_results['mean_background_iou']}")
print(f"\tMean Table IoU: {test_results['mean_table_iou']}")
print(f"\tMean Objects IoU: {test_results['mean_objects_iou']}")
print(f"\tMean IoU: {test_results['mean_iou']}")

----------------------------------------------------------------------------------------------------------------------------

test_results['annotations']['/data/tabletop_dataset/test_set_small/scene_00000/0']['metrics']

----------------------------------------------------------------------------------------------------------------------------

dl = data_loader.get_TOD_test_dataloader(TOD_filepath, data_loading_params, batch_size=10, num_workers=8, shuffle=False)
batch = next(dl.__iter__())
rgb_imgs = torch_to_numpy(batch['rgb']) # Shape: [N x H x W x 3]
xyz_imgs = torch_to_numpy(batch['xyz']) # Shape: [N x H x W x 3]
labels = torch_to_numpy(batch['labels']) # Shape: [N x H x W]
N, H, W = labels.shape[:3]

part_to_plot=0
plotting_batch_size = 10
image_begin = part_to_plot*plotting_batch_size
image_end = min((part_to_plot+1)*plotting_batch_size, N)
print("Plotting images: {0} to {1}".format(image_begin+1, image_end))

fg_only = True

fig_index = 1
for i in range(image_begin, image_end):
    
    fig = plt.figure(fig_index); fig_index += 1
    fig.set_size_inches(20,5)

    # Get prediction from test results
    key = batch['scene_dir'][i] + str(batch['view_num'][i].item())
    seg_mask = test_results['annotations'][key]['prediction']
    print(key)
    
    # Plot image
    plt.subplot(1,4,1)
    plt.imshow(rgb_imgs[i,...].astype(np.uint8))
    plt.title('Image {0}'.format(i+1))

    # Plot Depth
    plt.subplot(1,4,2)
    plt.imshow(xyz_imgs[i,...,2])
    plt.title('Depth')
    
    # Plot labels
    plt.subplot(1,4,3)
    if fg_only:
        gt_masks = labels[i,...].clip(0,2)
    else:
        gt_masks = labels[i,...]
    num_colors = max(np.unique(gt_masks).shape[0], np.unique(seg_mask).shape[0])
    plt.imshow(util_.get_color_mask(gt_masks, nc=num_colors))
    plt.title(f"GT Masks: {num_colors}")
    
    # Plot prediction
    plt.subplot(1,4,4)
    plt.imshow(util_.get_color_mask(seg_mask, nc=num_colors))
    plt.title(f"Predicted Masks: {np.unique(seg_mask).shape[0]}")
    
----------------------------------------------------------------------------------------------------------------------------

import evaluation

evaluation.batch_IoU(labels[0:1,...].clip(0,2)==2, np.expand_dims(test_results['annotations']['/data/tabletop_dataset/test_set_small/scene_00000/0']['prediction'],0)==2)

"""

def test_evaluation_multilabel_metrics():
	pass
""" test the correctness of evaluation.multilabel_metrics
import evaluation

gt = np.array([[2,2,2,3],
               [2,2,3,3,],
               [4,4,5,3],
               [4,5,5,5]])
pred = np.array([[6,6,4,5],
                 [7,6,4,5],
                 [3,3,2,5],
                 [3,3,2,2]])
metrics_dict = evaluation.multilabel_metrics(pred, gt)
assert metrics_dict['Objects F-measure'] == 0.75
assert metrics_dict['Objects Precision'] == 0.75
assert metrics_dict['Objects Recall'] == 0.75

gt = np.array([[2,2,0,0],
               [2,2,3,3,],
               [4,4,5,3],
               [0,0,5,5]])
metrics_dict = evaluation.multilabel_metrics(pred, gt)
assert np.isclose(metrics_dict['Objects Precision'], 0.625)
assert np.isclose(metrics_dict['Objects Recall'], 0.83333, atol=1e-5)
assert np.isclose(metrics_dict['Objects F-measure'], 0.71429, atol=1e-5)
"""

def add_noise_to_test_and_save():
	pass
"""
import sys, os
import json
from time import time
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import cv2

import data_augmentation, data_loader
import simulation.simulation_util as sim_util
import util.util as util_

NUM_VIEWS = 7

# Depth noise params
noise_params = {
    # Camera/Frustum parameters
    'img_width' : 640, 
    'img_height' : 480,
    'near' : 0.01,
    'far' : 100,
    'fov' : 60, # vertical field of view in angles

    # Multiplicative noise
    'gamma_shape' : 1000.,
    'gamma_scale' : 0.001,
    
    # Additive noise
    'gaussian_scale' : 0.01, # 1cm standard dev
    'gp_rescale_factor' : 4,
    
    # Random ellipse dropout
    'ellipse_dropout_mean' : 10, 
    'ellipse_gamma_shape' : 5.0, 
    'ellipse_gamma_scale' : 1.0,

    # Random high gradient dropout
    'gradient_dropout_left_mean' : 15, 
    'gradient_dropout_alpha' : 2., 
    'gradient_dropout_beta' : 5.,

    # Random pixel dropout
    'pixel_dropout_alpha' : 1., 
    'pixel_dropout_beta' : 10.,
}

test_set_filepath = '/data/tabletop_dataset_v3/test_set/'
test_dirs = sorted(os.listdir(test_set_filepath))

for i, tdir in enumerate(test_dirs):

    if i % 50 == 0:
        print(f"Working on {tdir}...")
    
    scene_dir = test_set_filepath + tdir + '/'
    
    for view_num in range(NUM_VIEWS):
        
        ### Load the depth image ###
        depth_img_filename = scene_dir + f"depth_{view_num:05d}.png"
        depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH) # Shape: [H x W]
        depth_img = (depth_img / 1000.).astype(np.float32) # millimeters -> meters
        seg_filename = scene_dir + f"segmentation_{view_num:05d}.png"
        seg_img = util_.imread_indexed(seg_filename)
        
        ### Add noise to it ###
        
        # add random noise to depth
        depth_img = data_augmentation.add_noise_to_depth(depth_img, noise_params)
        depth_img = data_augmentation.dropout_random_ellipses(depth_img, noise_params)
        # depth_img = data_augmentation.dropout_near_high_gradients(depth_img, seg_img, noise_params)
        depth_img = data_augmentation.dropout_random_pixels(depth_img, noise_params)

        # Compute xyz ordered point cloud
        xyz_img = data_loader.compute_xyz(depth_img, noise_params)
        xyz_img = data_augmentation.add_noise_to_xyz(xyz_img, depth_img, noise_params)

        # Save it
        depth_filename = scene_dir + f"depth_noisy_{view_num:05d}.png"
        cv2.imwrite(depth_filename, sim_util.saveable_depth_image(depth_img))

print('Done!')
"""

def method_to_call_encoders():
	pass
""" this code isn't that useful, but i'll save it here

	def apply_encoders(self, input_dict):
	    "" Apply encoders to input

	        @param input_dict: A dictionary of torch tensors of different modalities.
	                           e.g. keys could include: rgb, xyz, surface_normal
	    ""
	    output = {}

	    if self.params['use_rgb']:
	        output['rgb'] = self.encoders['rgb'](input_dict['rgb'].to(self.device))
	    if self.params['use_depth']:
	        output['xyz'] = self.encoders['xyz'](input_dict['xyz'].to(self.device))
	    if self.params['use_surface_normals']:
	        output['surface_normal'] = self.encoders['surface_normal'](input_dict['surface_normal'].to(self.device))

	    return output
"""

def check_if_tabletop_scene_is_compromised():
	pass
""" check to make sure all of the directories looks fine

# Checks that all scenes are there

import os

path_to_check = '/data/tabletop_dataset_v5/training_set/'
os.chdir(path_to_check)
temp = os.listdir('.')

temp1 = [int(x.split('_')[1]) for x in temp] 

num_scenes = 40000 # 2k for test
set(range(num_scenes)).difference(set(temp1))

----------------------------------------------------------------------------------------------------------------------------

# Checks that each scene has the right amount of stuff

import os

path_to_check = '/data/tabletop_dataset_v5/training_set/'
os.chdir(path_to_check)
temp = sorted(os.listdir('.'))

def check_dir(direc):
    num_views = 7
    return all([f'segmentation_{i:05d}.png' in direc for i in range(num_views)]) and \
           all([f'depth_{i:05d}.png' in direc for i in range(num_views)]) and \
           all([f'rgb_{i:05d}.jpeg' in direc for i in range(num_views)]) and \
           'scene_description.txt' in direc

for direct in temp:
    blah = os.listdir(direct)
    if not check_dir(blah):
        print(direct, blah)

"""

def compute_surface_normals_and_timeit():
	pass
""" Compute surface normals with open3d-python library, time it in Jupyter notebook

import open3d

----------------------------------------------------------------------------------------------------------------------------

# load depth image
depth_img_filename = '/data/tabletop_dataset/training_set/scene_00010/depth_00000.png'
depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH)
depth_img = (depth_img / 1000.).astype(np.float32)

----------------------------------------------------------------------------------------------------------------------------

# Create ordered point cloud
camera_params = {
               # Camera/Frustum parameters
               'img_width' : 640, 
               'img_height' : 480,
               'near' : 0.01,
               'far' : 100,
               'fov' : 60, # vertical field of view in angles
              }

def compute_xyz(depth_img, camera_params):

    # Compute focal length from camera parameters
    aspect_ratio = camera_params['img_width'] / camera_params['img_height']
    e = 1 / (np.tan(np.radians(camera_params['fov']/2.)))
    t = camera_params['near'] / e; b = -t
    r = t * aspect_ratio; l = -r
    alpha = camera_params['img_width'] / (r-l) # pixels per meter
    focal_length = camera_params['near'] * alpha # focal length of virtual camera (frustum camera)

    indices = util_.build_matrix_of_indices(camera_params['img_height'], camera_params['img_width'])
    indices[..., 0] = np.flipud(indices[..., 0]) # pixel indices start at top-left corner. for OpenGL, it starts at bottom-left
    z_e = depth_img
    x_e = (indices[..., 1] - camera_params['img_width']/2) * z_e / focal_length
    y_e = (indices[..., 0] - camera_params['img_height']/2) * z_e / focal_length
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    
    return xyz_img

xyz_img = compute_xyz(depth_img, camera_params)

----------------------------------------------------------------------------------------------------------------------------

# Reshape to unordered point cloud
unordered_pc = xyz_img.reshape(-1, 3)
pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(unordered_pc) 
pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) # Transform it so it's not upside down

----------------------------------------------------------------------------------------------------------------------------

# Compute surface normals with open3d
open3d.estimate_normals(pcd, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=15))
open3d.orient_normals_to_align_with_direction(pcd, orientation_reference=np.array([0,1,0])) # camera up vector is y-axis
surface_normals = np.asarray(pcd.normals).reshape(xyz_img.shape) # Shape: [H x W x 3]
# open3d.draw_geometries([pcd])

----------------------------------------------------------------------------------------------------------------------------

# Time the procedure. For a 640x480 image, it takes about .15 seconds. Not too bad
%timeit open3d.estimate_normals(pcd, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=15))

----------------------------------------------------------------------------------------------------------------------------

# downsample point cloud for better visualization
downpcd = open3d.voxel_down_sample(pcd, voxel_size=0.025)
open3d.estimate_normals(downpcd, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=15))
open3d.orient_normals_to_align_with_direction(downpcd, orientation_reference=np.array([0,1,0])) # camera up vector is y-axis
open3d.draw_geometries([downpcd])

----------------------------------------------------------------------------------------------------------------------------

##### COMPUTING SURFACE NORMALS USING MY FAST NUMPY CODE #####

# Calculate image normals with numpy
def calculate_surface_normals(xyz_img):
    xyz_img = xyz_img.astype(float)
    dzdx = np.roll(xyz_img, -1, axis=1) - xyz_img # Shape: [H x W x 3]
    dzdy = np.roll(xyz_img, -1, axis=0) - xyz_img # Shape: [H x W x 3]
    cross_product = np.cross(dzdx, dzdy, axis=2)
    surface_normals = cross_product / np.linalg.norm(cross_product, axis=2, keepdims=True)
    # NOTE: This does not take care of corners, but that's okay
    return surface_normals

surface_normals = calculate_surface_normals(xyz_img)

----------------------------------------------------------------------------------------------------------------------------

# Time the procedure. For a 640x480 image, it takes about .03 seconds. About 5x faster than open3d
%timeit calculate_surface_normals(xyz_img)

----------------------------------------------------------------------------------------------------------------------------

# Reshape to unordered point cloud
xyz_img = compute_xyz(depth_img, camera_params)
unordered_pc = xyz_img.reshape(-1, 3)
pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(unordered_pc) 
pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) # Transform it so it's not upside down
pcd.normals = open3d.Vector3dVector(calculate_surface_normals(xyz_img).reshape(-1,3))
open3d.orient_normals_to_align_with_direction(downpcd, orientation_reference=np.array([0,1,0])) # camera up vector is y-axis

----------------------------------------------------------------------------------------------------------------------------

# downsample point cloud for better visualization
downscale_factor = 6
new_height = 480 // downscale_factor; new_width = 640 // downscale_factor;
resized_depth_img = cv2.resize(depth_img, (new_width, new_height))
small_camera_params = camera_params.copy()
small_camera_params['img_height'] = new_height
small_camera_params['img_width'] = new_width
xyz_img = compute_xyz(resized_depth_img, small_camera_params)

unordered_pc = xyz_img.reshape(-1, 3)
pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(unordered_pc) 
pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) # Transform it so it's not upside down
pcd.normals = open3d.Vector3dVector(calculate_surface_normals(xyz_img).reshape(-1,3))
# open3d.estimate_normals(pcd, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=15))
open3d.orient_normals_to_align_with_direction(pcd, orientation_reference=np.array([0,1,0])) # camera up vector is y-axis

open3d.draw_geometries([pcd])
"""

def draw_point_cloud_from_real_RGBD_images():
	pass
"""

# Call this after plotting in this section: "Evaluate on real RGBD images"

i = 7
xyz_img = xyz_imgs[i]
unordered_pc = xyz_img.reshape(-1, 3)
pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(unordered_pc) 
unordered_colors = (util_.get_color_mask(labels[i,...]) / 255.).reshape(-1, 3)
pcd.colors = open3d.Vector3dVector(unordered_colors)
pcd.transform([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]]) # Transform it so it's not upside down
open3d.estimate_normals(pcd, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=15))
open3d.orient_normals_to_align_with_direction(pcd, orientation_reference=np.array([0,1,0])) # camera up vector is y-axis
open3d.draw_geometries([pcd])

"""

def visually_test_depth_data_augmentation():
	pass
"""

scene_dir = 'scene_00070'
view_num = 0

rgb_img_filename = f'/data/tabletop_dataset/training_set/{scene_dir}/rgb_{view_num:05d}.jpeg'
rgb_img = cv2.cvtColor(cv2.imread(rgb_img_filename), cv2.COLOR_BGR2RGB)

depth_img_filename = f'/data/tabletop_dataset/training_set/{scene_dir}/depth_{view_num:05d}.png'
depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH)
depth_img = (depth_img / 1000.).astype(np.float32)

seg_img_filename = f'/data/tabletop_dataset/training_set/{scene_dir}/segmentation_{view_num:05d}.png'
seg_img = util_.imread_indexed(seg_img_filename)

fig = plt.figure(1)
fig.set_size_inches(15,5)

plt.subplot(1,3,1)
plt.imshow(rgb_img)

plt.subplot(1,3,2)
plt.imshow(depth_img)

plt.subplot(1,3,3)
plt.imshow(seg_img)

----------------------------------------------------------------------------------------------------------------------------

params = {'gradient_dropout_left_mean' : 15, 
          'gradient_dropout_alpha' : 2., 
          'gradient_dropout_beta' : 5.}
gradient_dropout_depth_img = data_augmentation.dropout_near_high_gradients(depth_img, seg_img, params)
plt.imshow(gradient_dropout_depth_img)

----------------------------------------------------------------------------------------------------------------------------

params = {'ellipse_dropout_mean' : 10, 
          'ellipse_gamma_shape' : 5.0, 
          'ellipse_gamma_scale' : 1.0}
ellipse_dropout_depth_img = data_augmentation.dropout_random_ellipses(depth_img, params)
plt.imshow(ellipse_dropout_depth_img)

----------------------------------------------------------------------------------------------------------------------------

params = {'pixel_dropout_alpha' : 1., 
          'pixel_dropout_beta' : 10.}
pixel_dropout_depth_img = data_augmentation.dropout_random_pixels(depth_img, params)
plt.imshow(pixel_dropout_depth_img)

"""

def real_RGBD_images_camera_params():
	pass
""" This is the camera parameters for the 70 photos I took (using 2 different cameras)

import json 

real_rgbd_images_filepath = '/data/tabletop_dataset/real_RGBD_images/'
real_rgbd_images_params = {
    # Camera intrinsics parameters
    'img_width' : 640, 
    'img_height' : 480,
    'fx' : 612.937, # 614.368
    'fy' : 613.173, # 614.33
    'x_offset' : 322.549, # 319.487
    'y_offset' : 248.158, # 241.57
    # Note: the current camera parameters are for last 20 images. the commented ones are for first 50.
    #           first 50 / last 20
    #       fx: 614.368  / 612.937
    #       fy: 614.33   / 613.173
    # x_offset: 319.487  / 322.549
    # y_offset: 241.57   / 248.158
}

camera_params_filename = real_rgbd_images_filepath + 'camera_params.json'
with open(camera_params_filename, 'w') as save_file:  
    json.dump(real_rgbd_images_params, save_file)
"""

def refinement_module_no_batch():
	pass
"""
def forward(self, features, probs):
    "" Compute forward pass of network. NON BATCH WISE

        @param features: [C x H x W] torch.FloatTensor
        @param probs: [K x H x W] torch.FloatTensor, where K = num_classes. Assumes probs is in [0,1], softmax'ed outside of this method

        @return [C x H x W] torch.FloatTensor of updated features
    ""
    _, H, W = features.shape
    
    # Per-class weighted average pooling
    weighted_features = features.unsqueeze(0) * probs.unsqueeze(1) # Shape: [K x C x H x W]
    weight_sums = torch.sum(probs, dim=(1,2), keepdim=True) # Shape: [K x 1 x 1]
    weighted_avgpool_features = torch.sum(weighted_features / weight_sums.unsqueeze(1), 
                                          dim=(2,3), keepdim=True) # Shape: [K x C x 1 x 1]

    # Tile (Broadcasting) and subtract
    compared_features = features.unsqueeze(0) - weighted_avgpool_features # Shape: [K x C x H x W]

    # Conv layers (treat K as batch size. Then same filter is applied for every class)
    out = self.conv1(compared_features) # Shape: [K x C x H x W]
    out = self.conv2(torch.reshape(out, (1, self.fd*self.nc, H, W))) # Shape: [1 x C x H x W]

    return out
"""

def refinement_dropout_using_beta_distribution():
	pass
"""
class Refinement_Dropout(nn.Module):

    def __init__(self, alpha, beta):
        super(Refinement_Dropout, self).__init__()
        self.beta_dist = torch.distributions.Beta(alpha, beta)

    def forward(self, X):
        if self.training:
            p = self.beta_dist.sample()
            X = F.dropout(X, p=p) * (1/(1-p))
        return X

"""

def refinement_stuff():
	pass
""" 
go to git commit fcad0e to see all the code that's there
go to git commit f18eb4 to see everything refinement-related that I removed
"""

def fun_test_to_optimize_cross_product():
    pass
"""
a = torch.randn(3, dtype=torch.float, requires_grad=True)
print(f"a init: {a}")
print(f"a init normalized: {a / torch.norm(a)}")
b = torch.randn(3, dtype=torch.float, requires_grad=True)
print(f"b init: {b}")
print(f"b init normalized: {b / torch.norm(b)}")

optimizer = torch.optim.SGD([a, b], 0.01)

losses = []
for i in range(1000):
    
    x = torch.cross(a,b)
    loss = torch.sum(x**2)
    
    losses.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
plt.plot(losses)
print(a / torch.norm(a))
print(b / torch.norm(b))

# a should now be parallel to b. magnitude of a doesn't matter
"""

def transform_to_table_coordinates():
    pass
"""
data_loader = reload(data_loader)
data_augmentation = reload(data_augmentation)
dl = data_loader.get_TOD_train_dataloader(TOD_filepath, data_loading_params, batch_size=8, num_workers=8, shuffle=True)

# try without noise
# temp = data_loading_params.copy()
# temp['use_data_augmentation'] = False
# dl = data_loader.get_TOD_train_dataloader(TOD_filepath, temp, batch_size=8, num_workers=8, shuffle=True)

dl_iter = dl.__iter__()

----------------------------------------------------------------------------------------------------------------------------

batch = next(dl_iter)
rgb_imgs = torch_to_numpy(batch['rgb'], is_standardized_image=True) # Shape: [N x H x W x 3]
xyz_imgs = torch_to_numpy(batch['xyz']) # Shape: [N x H x W x 3]
labels = torch_to_numpy(batch['labels']) # Shape: [N x H x W]
surface_normals = torch_to_numpy(batch['surface_normal']) # Shape: [N x H x W x 3]
N, H, W = labels.shape[:3]

----------------------------------------------------------------------------------------------------------------------------

im_to_show = 0

fig = plt.figure(1)
fig.set_size_inches(20,5)

# Plot image
plt.subplot(1,4,1)
plt.imshow(rgb_imgs[im_to_show,...].astype(np.uint8))
plt.title('RGB')

# Plot Depth
plt.subplot(1,4,2)
plt.imshow(xyz_imgs[im_to_show,...,2])
plt.title('Depth')

# Plot labels
plt.subplot(1,4,3)
gt_masks = labels[im_to_show,...]
plt.imshow(util_.get_color_mask(gt_masks))
plt.title(f"GT Masks: {np.unique(gt_masks).shape[0]}")

----------------------------------------------------------------------------------------------------------------------------

# Get stuff
rgb_im = rgb_imgs[im_to_show]
xyz_im = xyz_imgs[im_to_show]
surface_normal_im = surface_normals[im_to_show]
label_im = labels[im_to_show]

----------------------------------------------------------------------------------------------------------------------------

# Compute average surface normal with weighted average pooling
tabletop_mask = label_im == 1
nonzero_depth_mask = ~np.isclose(xyz_im[...,2], 0)
tabletop_mask = np.logical_and(tabletop_mask, nonzero_depth_mask)

# inflate tabletop_mask so that surface normal computation is correct. we do this because of how surface normal is computed
tabletop_mask = np.logical_and(tabletop_mask, np.roll(nonzero_depth_mask, -1, axis=0))
tabletop_mask = np.logical_and(tabletop_mask, np.roll(nonzero_depth_mask, 1,  axis=0))
tabletop_mask = np.logical_and(tabletop_mask, np.roll(nonzero_depth_mask, -1, axis=1))
tabletop_mask = np.logical_and(tabletop_mask, np.roll(nonzero_depth_mask, 1,  axis=1))

# via indexing
table_y = np.mean(surface_normal_im[tabletop_mask], axis=0)
table_y = table_y / (np.linalg.norm(table_y) + 1e-10)
print(table_y)

# via weighted average pooling. NOTE: this is differentiable w.r.t. tabletop_mask
table_y = np.sum(surface_normal_im * np.expand_dims(tabletop_mask, axis=2), axis=(0,1)) / np.sum(tabletop_mask)
table_y = table_y / (np.linalg.norm(table_y) + 1e-10)
print(table_y)

----------------------------------------------------------------------------------------------------------------------------

# Project camera z-axis onto table plane. NOTE: this is differentiable w.r.t. table_y
camera_z = np.array([0,0,1]).astype(np.float32)
table_z = camera_z - np.dot(table_y, camera_z) * table_y
table_z = table_z / (np.linalg.norm(table_z) + 1e-10)
print(table_z)

----------------------------------------------------------------------------------------------------------------------------

# Get table x-axis. NOTE: this is differentiable w.r.t. table_y, table_z, since cross products are differentiable
# Another note: cross product adheres to the handedness of the coordinate system, which is a left-handed system
table_x = np.cross(table_y, table_z)
table_x = table_x / (np.linalg.norm(table_x) + 1e-10)
print(table_x)

----------------------------------------------------------------------------------------------------------------------------

# Transform xyz depth map to table coordinates

# via indexing
table_mean = np.mean(xyz_imgs[im_to_show, tabletop_mask, ...], axis=0)
print(table_mean)

# via weighted average pooling
table_mean = np.sum(xyz_im * np.expand_dims(tabletop_mask, axis=2), axis=(0,1)) / np.sum(tabletop_mask)
print(table_mean)

x_projected = np.dot(xyz_im - table_mean, table_x)
y_projected = np.dot(xyz_im - table_mean, table_y)
z_projected = np.dot(xyz_im - table_mean, table_z)

----------------------------------------------------------------------------------------------------------------------------

### Plot RGB/Depth/Seg ###
fig = plt.figure(1)
fig.set_size_inches(20,5)

# Plot image
plt.subplot(1,4,1)
plt.imshow(rgb_imgs[im_to_show,...].astype(np.uint8))
plt.title('RGB')

# Plot Depth
plt.subplot(1,4,2)
plt.imshow(xyz_imgs[im_to_show,...,2])
plt.title('Depth')

# Plot labels
plt.subplot(1,4,3)
gt_masks = labels[im_to_show,...]
plt.imshow(util_.get_color_mask(gt_masks))
plt.title(f"GT Masks: {np.unique(gt_masks).shape[0]}")




### Plot projected depth ###
fig = plt.figure(2)
fig.set_size_inches(20,5)

# Plot image
plt.subplot(1,4,1)
plt.imshow(x_projected)
plt.title('x_projected')

# Plot Depth
plt.subplot(1,4,2)
plt.imshow(y_projected)
plt.title('y_projected')

# Plot labels
plt.subplot(1,4,3)
plt.imshow(z_projected)
plt.title("z_projected")




### Plot values near 0 to double check that this is right ###
fig = plt.figure(3)
fig.set_size_inches(20,5)

# Plot image
plt.subplot(1,4,1)
plt.imshow(np.isclose(x_projected, 0, atol=1e-2))
plt.title('x_projected near 0')

# Plot Depth
plt.subplot(1,4,2)
plt.imshow(np.isclose(y_projected, 0, atol=2e-2))
plt.title('y_projected near 0')

# Plot labels
plt.subplot(1,4,3)
plt.imshow(np.isclose(z_projected, 0, atol=1e-2))
plt.title("z_projected near 0")
"""

def test_table_xyz_transform():
	pass
"""
data_loader = reload(data_loader)
data_augmentation = reload(data_augmentation)
dl = data_loader.get_TOD_train_dataloader(TOD_filepath, data_loading_params, batch_size=8, num_workers=8, shuffle=True)

# try without noise
# temp = data_loading_params.copy()
# temp['use_data_augmentation'] = False
# dl = data_loader.get_TOD_train_dataloader(TOD_filepath, temp, batch_size=8, num_workers=8, shuffle=True)

dl_iter = dl.__iter__()

----------------------------------------------------------------------------------------------------------------------------

batch = next(dl_iter)
rgb_imgs = torch_to_numpy(batch['rgb'], is_standardized_image=True) # Shape: [N x H x W x 3]
xyz_imgs = torch_to_numpy(batch['xyz']) # Shape: [N x H x W x 3]
labels = torch_to_numpy(batch['labels']) # Shape: [N x H x W]
surface_normals = torch_to_numpy(batch['surface_normal']) # Shape: [N x H x W x 3]
N, H, W = labels.shape[:3]

----------------------------------------------------------------------------------------------------------------------------

im_to_show = 0

fig = plt.figure(1)
fig.set_size_inches(20,5)

# Plot image
plt.subplot(1,4,1)
plt.imshow(rgb_imgs[im_to_show,...].astype(np.uint8))
plt.title('RGB')

# Plot Depth
plt.subplot(1,4,2)
plt.imshow(xyz_imgs[im_to_show,...,2])
plt.title('Depth')

# Plot labels
plt.subplot(1,4,3)
gt_masks = labels[im_to_show,...]
plt.imshow(util_.get_color_mask(gt_masks))
plt.title(f"GT Masks: {np.unique(gt_masks).shape[0]}")

----------------------------------------------------------------------------------------------------------------------------

# Get stuff
rgb_im = rgb_imgs[im_to_show]
xyz_im = xyz_imgs[im_to_show]
surface_normal_im = surface_normals[im_to_show]
label_im = labels[im_to_show]

----------------------------------------------------------------------------------------------------------------------------

util_ = reload(util_)
tabletop_mask = batch['labels'][im_to_show] == 1
table_xyz = util_.transform_camera_xyz_to_table_xyz(batch['xyz'][im_to_show].cuda(),
                                                batch['surface_normal'][im_to_show].cuda(),
                                                tabletop_mask.cuda()
                                               )

----------------------------------------------------------------------------------------------------------------------------

### Plot RGB/Depth/Seg ###
fig = plt.figure(1)
fig.set_size_inches(20,5)

# Plot image
plt.subplot(1,4,1)
plt.imshow(rgb_imgs[im_to_show,...].astype(np.uint8))
plt.title('RGB')

# Plot Depth
plt.subplot(1,4,2)
plt.imshow(xyz_imgs[im_to_show,...,2])
plt.title('Depth')

# Plot labels
plt.subplot(1,4,3)
gt_masks = labels[im_to_show,...]
plt.imshow(util_.get_color_mask(gt_masks))
plt.title(f"GT Masks: {np.unique(gt_masks).shape[0]}")


x_projected = table_xyz[0].cpu().numpy()
y_projected = table_xyz[1].cpu().numpy()
z_projected = table_xyz[2].cpu().numpy()


### Plot projected depth ###
fig = plt.figure(2)
fig.set_size_inches(20,5)

# Plot image
plt.subplot(1,4,1)
plt.imshow(x_projected)
plt.title('x_projected')

# Plot Depth
plt.subplot(1,4,2)
plt.imshow(y_projected)
plt.title('y_projected')

# Plot labels
plt.subplot(1,4,3)
plt.imshow(z_projected)
plt.title("z_projected")




### Plot values near 0 to double check that this is right ###
fig = plt.figure(3)
fig.set_size_inches(20,5)

# Plot image
plt.subplot(1,4,1)
plt.imshow(np.isclose(x_projected, 0, atol=1e-2))
plt.title('x_projected near 0')

# Plot Depth
plt.subplot(1,4,2)
plt.imshow(np.isclose(y_projected, 0, atol=1e-2))
plt.title('y_projected near 0')

# Plot labels
plt.subplot(1,4,3)
plt.imshow(np.isclose(z_projected, 0, atol=1e-2))
plt.title("z_projected near 0")
"""

def visualize_center_predictions():
    pass
"""

sys.path.append('/home/chrisxie/projects/rar/src/util/')
import flowlib

fig_index = 1
for i in range(N):
    
    fig = plt.figure(fig_index); fig_index += 1
    fig.set_size_inches(20,5)

    # Plot image
    plt.subplot(1,4,1)
    plt.imshow(rgb_imgs[i,...].astype(np.uint8))
    plt.title('Image {0}'.format(i+1))

    # Plot Depth
    plt.subplot(1,4,2)
    plt.imshow(xyz_imgs[i,...,2])
    plt.title('Depth')
    
    # Plot Foreground labels
    plt.subplot(1,4,3)
    gt_masks = foreground_labels[i,...]
    plt.imshow(util_.get_color_mask(gt_masks))
    plt.title(f"GT Masks: {np.unique(gt_masks).shape[0]}")
    
    # Plot Direction labels
    plt.subplot(1,4,4)
    plt.imshow(flowlib.flow_to_image(direction_labels[i,...,:2]))
    plt.title(f"Center Directions")

"""

def visualize_hough_maps_on_cabinet():
    pass
""" Visualize cabinet photo that Arsalan gave me

d = np.load('/home/chrisxie/cabinet_D415.npy', encoding='bytes').item()

rgb_img = d[b'rgb']
depth_img = d[b'depth']

# millimeters -> meters
depth_img = (depth_img / 1000.).astype(np.float32)

# Compute xyz ordered point cloud
xyz_img = data_loader.compute_xyz(depth_img, dl.dataset.params)
xyz_img = torch.from_numpy(xyz_img).permute(2,0,1).unsqueeze(0)

----------------------------------------------------------------------------------------------------------------------------

seg_network.eval_mode()

### Compute segmentation masks ###
st_time = time()
seg_masks, direction_predictions = seg_network.run_on_batch({'xyz' : xyz_img})
total_time = time() - st_time
print('Total time taken for Segmentation: {0} seconds'.format(round(total_time, 3)))
print('FPS: {0}'.format(round(N / total_time,3)))

st_time = time()
hough_directions, pixel_locations, num_maxes = hv_layer((seg_masks == 2).int(), direction_predictions.flip([1]))
# Note: need foreground (object)/background for masks, and directions need to be (x,y), not (y,x)
total_time = time() - st_time
print('Total time taken for Hough Transform: {0} seconds'.format(round(total_time, 3)))
print('FPS: {0}'.format(round(N / total_time,3)))

# Get segmentation masks in numpy
seg_masks = seg_masks.cpu().numpy()
direction_predictions = direction_predictions.cpu().numpy().transpose(0,2,3,1)
hough_directions = hough_directions.cpu().numpy()
pixel_locations = pixel_locations.cpu().numpy()
num_maxes = num_maxes.cpu().numpy()

----------------------------------------------------------------------------------------------------------------------------

fig = plt.figure();
fig.set_size_inches(20,5)

# Plot image
plt.subplot(1,6,1)
plt.imshow(rgb_img)
plt.title('Image')

# Plot Depth
plt.subplot(1,6,2)
plt.imshow(xyz_img[0,2,...])
plt.title('Depth')

# Plot prediction
plt.subplot(1,6,3)
plt.imshow(util_.get_color_mask(seg_masks[0,...]))
plt.title(f"Predicted Masks: {np.unique(seg_masks[0,...]).shape[0]}")

# Plot Center Direction Predictions
plt.subplot(1,6,4)
fg_mask = np.expand_dims(seg_masks[0,...] == 2, axis=-1)
plt.imshow(flowlib.flow_to_image(direction_predictions[0,...] * fg_mask))
plt.title("Center Direction Predictions")

# # Plot Hough Map
# plt.subplot(1,6,5)
# plt.imshow(hough_maps[0])
# plt.title(f"Hough Map. Min: {hough_maps[0].min()}, Max: {hough_maps[0].max()}")

# # Plot Hough Directions
# plt.subplot(1,6,6)
# plt.imshow(hough_directions[0])
# plt.title(f"Hough Directions. Min: {hough_directions[0].min()}, Max: {hough_directions[0].max()}")

# Plot Hough Directions
plt.subplot(1,6,5)
plt.imshow(hough_directions[0])
plt.title(f"Hough Directions. Max: {hough_directions[0].max():.2f}")

# Plot Object Centers
plt.subplot(1,6,6)
object_center_plot = np.zeros_like(hough_directions[0]) # Shape: [H x W]
index_array = pixel_locations[0][:, :num_maxes[0]]
object_center_plot[index_array[0], index_array[1]] = 1
object_center_plot = cv2.GaussianBlur(object_center_plot, ksize=(29,29), sigmaX=0)
plt.imshow(object_center_plot)
plt.title("Num Object Centers: {}".format(num_maxes[0]))

"""

def nice_numpy_printing():
    pass
"""
np.set_printoptions(precision=4, linewidth=150)
"""

def visualize_segmentations_nicely():
    pass
"""
img_num = 0

------------------------------------------------------------------------------------------------

for i in range(N):
    save_filename = '/home/chrisxie/Pictures/predictions_' + str(img_num) + '.png'
    util_.visualize_segmentation(rgb_imgs[i,...].astype(np.uint8), seg_masks[i,...], save_dir=save_filename)
    img_num += 1
"""

def compare_IMP_stuff_visual():
    pass
"""
no_imp_dir = '/home/chrisxie/Pictures/no_IMP/'
oc_dir = '/home/chrisxie/Pictures/openclose/'
oc_ccc_dir = '/home/chrisxie/Pictures/openclose_ccc/'
oc_ccc_tr_dir = '/home/chrisxie/Pictures/openclose_ccc_table_RANSAC/'

fig_index = 1
total_subplots = 4

oc_ccc_help = [1,2]
oc_ccc_tr_help = [3,13,24,27]
accurate_masks = [10,38,41,45]
inaccurate_masks = [11,32]
ccc_help_after_oc = [17,26]
for_fun_cabinet_photos = [1,5,10] # img_filename = 'cabinet_predictions_' + str(i) + '.png'

for i in for_fun_cabinet_photos:
    
    img_filename = 'predictions_' + str(i) + '.png'
    
    fig = plt.figure(fig_index); fig_index += 1
    fig.set_size_inches(20,5)
    
    plt.subplot(1,4,1)
    plt.imshow(cv2.cvtColor(cv2.imread(no_imp_dir + img_filename), cv2.COLOR_BGR2RGB))
    plt.title(f'No IMP. Image {i}')
    
    plt.subplot(1,4,2)
    plt.imshow(cv2.cvtColor(cv2.imread(oc_dir + img_filename), cv2.COLOR_BGR2RGB))
    plt.title('+Open/Close')    
    
    plt.subplot(1,4,3)
    plt.imshow(cv2.cvtColor(cv2.imread(oc_ccc_dir + img_filename), cv2.COLOR_BGR2RGB))
    plt.title('+Open/Close+CCC')
    
    plt.subplot(1,4,4)
    plt.imshow(cv2.cvtColor(cv2.imread(oc_ccc_tr_dir + img_filename), cv2.COLOR_BGR2RGB))
    plt.title('+Open/Close+CCC+RANSAC')
"""

def process_OSD_dataset():
    pass
""" Process this dataset

# Issue: Open3D removes NaNs from point cloud
# Solution: Process with pypcd (using python 2.7), change NaNs to 0, write it back out

# From directory: /data/OSD/OSD-0.2/pcd
# NOTE: need python2. on patillo, use "conda activate py2"


import os
import pypcd
import numpy as np

pcd_files = sorted(os.listdir('.'))
for pcd_file in pcd_files:

    # load pcd file
    pcd = pypcd.PointCloud.from_path(pcd_file)

    # Correct it (NaNs -> 0)
    pcd.pc_data['x'][np.isnan(pcd.pc_data['x'])] = 0
    pcd.pc_data['y'][np.isnan(pcd.pc_data['y'])] = 0
    pcd.pc_data['z'][np.isnan(pcd.pc_data['z'])] = 0

    # Write it back
    pcd.save_pcd(pcd_file, compression='binary_compressed')


"""

def look_at_depth_histograms_OCID():
    pass
"""
hists_per_row = 4
for i in range(int(np.ceil(N / hists_per_row))):
    
    fig = plt.figure(fig_index); fig_index += 1
    fig.set_size_inches(20,5)
    
    for j in range(hists_per_row):
        
        img_index = i*hists_per_row + j
        if img_index >= N:
            break
        
        z_vals = xyz_imgs[img_index, ..., 2]
        min_depth = z_vals[z_vals > 0.1].min()
        if min_depth < 0.75:
            b_or_t = 'Bottom'
        else:
            b_or_t = 'Top'
        
        plt.subplot(1, hists_per_row, j+1)
        plt.hist(z_vals.flatten(), bins=20)
        plt.title(f'Image {img_index+1}, Min Depth: {min_depth:0.2f}, {b_or_t}')
"""

def imwrite_indexed_OCID_labels():
    pass
"""

import os
import cv2
import glob
import numpy as np
from PIL import Image

dir_to_investigate = '/data/OCID-dataset/ARID10/table/bottom/mixed/seq13/label/'
save_dir = '/home/chrisxie/temp/'
label_filenames = sorted(glob.glob(dir_to_investigate + '*.png'))

def imwrite_indexed(filename,array):
    # Save indexed png with palette

    palette_abspath = '/data/tabletop_dataset_v5/palette.txt' # hard-coded filepath
    color_palette = np.loadtxt(palette_abspath, dtype=np.uint8).reshape(-1,3)

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

for label_filename in label_filenames:

    # Load the label in
    temp_idx = label_filename.split('/').index('OCID-dataset') # parse something like this: /data/OCID-dataset/YCB10/table/top/curved/seq36/pcd/result_2018-08-24-15-13-13.pcd
    subset = label_filename.split('/')[temp_idx+1] # one of: [ARID10, ARID20, YCB10]
    supporting_plane = 'floor' if 'floor' in label_filename else 'table'
    height = 'bottom' if 'bottom' in label_filename else 'top'

    label_img = cv2.imread(label_filename, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    if supporting_plane == 'table': # Merge 0 (NaNs) and 1 (non-table/object)
        label_img[label_img == 1] = 0
        label_img = (label_img.astype(int) - 1).clip(min=0).astype(np.uint8)

    # Remove table label
    label_img[label_img == 1] = 0

    # Write label out
    save_filename = save_dir + label_filename.split('/')[-1]
    imwrite_indexed(save_filename, label_img)

"""

def imwrite_indexed_OSD_labels():
    pass
"""

import os
import cv2
import glob
import numpy as np
from PIL import Image

dir_to_investigate = '/data/OSD/OSD-0.2-depth/annotation/'
save_dir = '/home/chrisxie/temp/'
label_filenames = sorted(glob.glob(dir_to_investigate + '*.png'))

def imwrite_indexed(filename,array):
    # Save indexed png with palette

    palette_abspath = '/data/tabletop_dataset_v5/palette.txt' # hard-coded filepath
    color_palette = np.loadtxt(palette_abspath, dtype=np.uint8).reshape(-1,3)

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

for label_filename in label_filenames:

    # Load the label in
    label_img = cv2.imread(label_filename, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    label_img[label_img > 0] = label_img[label_img > 0] + 1 # so values are in [0, 2, 3, ...] (e.g. no table label)

    # Remove table label
    label_img[label_img == 1] = 0

    # Write label out
    save_filename = save_dir + label_filename.split('/')[-1]
    imwrite_indexed(save_filename, label_img)

"""

def run_open_close_morphology_on_labels():
    pass
"""

# On patillo.cs.washington.edu
# Run this from /home/chrisxie/projects/ssc/

import os
import numpy as np
import cv2
import util.util as util_

# OCID
results_dir = '/home/chrisxie/projects/ssc/external/OCID_results/'
gt_dir = '/data/OCID-dataset/'
dataset_name = 'OCID'

# OSD
# results_dir = '/home/chrisxie/projects/ssc/external/OSD_results/'
# gt_dir = '/data/OSD/'
# dataset_name = 'OSD'

# Parse starts file
starts_file = gt_dir + 'label_files.txt'
f = open(starts_file, 'r')
label_filenames = [x.strip() for x in f.readlines()]

# Directories to change
dirs_to_change = [x for x in os.listdir(results_dir) if 'RRN_v5' in x]

for direc in dirs_to_change:

    prediction_dir = results_dir + direc + '/'
    print(f'Working on {prediction_dir}...')

    for label_filename in label_filenames:

        # Load the prediction (reverse the process I used to go from pcd_files.txt to label_files.txt)
        if dataset_name == 'OCID':
            pred_filename = label_filename.replace(gt_dir, prediction_dir)
            pred_filename = pred_filename.replace('/label/', '/pcd/')
        elif dataset_name == 'OSD':
            pred_filename = label_filename.replace(gt_dir, prediction_dir)
            pred_filename = pred_filename.replace('/OSD-0.2-depth/', '/OSD-0.2/')
            pred_filename = pred_filename.replace('/annotation/', '/pcd/')
        pred_img = util_.imread_indexed(pred_filename)

        # Run the open/close stuff

        # Get object ids. Remove background (0)
        obj_ids = np.unique(pred_img)
        if obj_ids[0] == 0:
            obj_ids = obj_ids[1:]

        # For each object id, open/close the masks
        for obj_id in obj_ids:
            mask = (pred_img == obj_id) # Shape: [H x W]

            ksize = 9
            opened_mask = cv2.morphologyEx(mask.astype(np.uint8), 
                                           cv2.MORPH_OPEN, 
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))
            opened_closed_mask = cv2.morphologyEx(opened_mask,
                                                  cv2.MORPH_CLOSE,
                                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))

            h_idx, w_idx = np.nonzero(mask)
            pred_img[h_idx, w_idx] = 0
            h_idx, w_idx = np.nonzero(opened_closed_mask)
            pred_img[h_idx, w_idx] = obj_id

        # Save to file
        util_.imwrite_indexed(pred_filename, pred_img)

"""

def run_RRN_v5_on_Mask_RCNN_OCID_labels():
    pass
"""
rgb_refinement_network = segmentation.RGBRefinementNetwork(None, rrn_params)
checkpoint_dir = '/home/chrisxie/projects/ssc/checkpoints/'
rgb_refinement_network.load(checkpoint_dir + 'RGBRefinementNetwork_iter100000_TableTop_v5_64c_checkpoint.pth.tar')

-----------------------------------------------------------------------------------------------------------------

import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm
    
H = 480
W = 640
    
ocid_filepath = datasets_base_dir + 'OCID-dataset/'
starts_file = ocid_filepath + 'label_files.txt'
prediction_dir = '/home/chrisxie/projects/ssc/external/OCID_results/Mask_RCNN/RGBD/'
save_dir = '/home/chrisxie/projects/ssc/external/OCID_results/Mask_RCNN/RGBD_RRN_v5/'

# Parse starts file
f = open(starts_file, 'r')
label_filenames = [x.strip() for x in f.readlines()]

# Run the RRN for each thing
for label_filename in tqdm(label_filenames):

    # Load the prediction (reverse the process I used to go from pcd_files.txt to label_files.txt)
    pred_filename = label_filename.replace(ocid_filepath, prediction_dir)
    pred_filename = pred_filename.replace('/label/', '/pcd/')
    pred_img = util_.imread_indexed(pred_filename)
    initial_masks = torch.from_numpy(pred_img).to('cuda') # Shape: [H x W]
    
    ### Load RGB/Depth ###
    pcd_filename = label_filename.replace('/label/', '/pcd/').replace('.png', '.pcd')
    print(pcd_filename)

    # Process .pcd file 
    temp_idx = pcd_filename.split('/').index('OCID-dataset') # parse something like this: /data/OCID-dataset/YCB10/table/top/curved/seq36/pcd/result_2018-08-24-15-13-13.pcd
    label_abs_path = '/'.join(pcd_filename.split('/')[temp_idx+1:])
    point_cloud = open3d.read_point_cloud(pcd_filename)
    
    # Fill in missing pixel values for RGB
    num_missing = 480*640 - np.asarray(point_cloud.colors).shape[0]
    filled_in_rgb_img = np.concatenate([np.asarray(point_cloud.colors), np.zeros((num_missing,3))])
    rgb_img = np.round(255 * filled_in_rgb_img.reshape(480,640,3)).astype(np.uint8)
    rgb_img = rgb_img.astype(np.float32)
    rgb_img = data_augmentation.BGR_image(rgb_img)
    rgb_img = data_augmentation.array_to_tensor(rgb_img).to('cuda') # Shape: [3 x H x W]
    
    # Fill in missing xyz values
    num_missing = 480*640 - np.asarray(point_cloud.points).shape[0]
    filled_in_points = np.concatenate([np.asarray(point_cloud.points), np.zeros((num_missing,3))])
    xyz_img = np.asarray(filled_in_points).reshape(480,640,3)
    xyz_img[np.isnan(xyz_img)] = 0
    xyz_img = data_augmentation.array_to_tensor(xyz_img).to('cuda') # Shape: [3 x H x W]
    
    ##### Run the RRN #####
    
    # Data structure to hold everything at end
    refined_masks = torch.zeros_like(initial_masks)

    # Dictionary to save crop indices
    crop_indices = {}

    mask_ids = torch.unique(initial_masks)
    if mask_ids[0] == 0: # Get rid of background
        mask_ids = mask_ids[1:]
    if len(mask_ids) > 0 and mask_ids[0] == 1: # Get rid of table
        mask_ids = mask_ids[1:]
    rgb_crops = torch.zeros((mask_ids.shape[0], 3, 224, 224), device='cuda')
    mask_crops = torch.zeros((mask_ids.shape[0], 224, 224), device='cuda')

    for index, mask_id in enumerate(mask_ids):
        mask = (initial_masks == mask_id).float() # Shape: [H x W]

        # crop the masks/rgb to 224x224 with some padding, save it as "initial_masks"
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(mask)
        x_padding = torch.round((x_max - x_min).float() * 0.25).item()
        y_padding = torch.round((y_max - y_min).float() * 0.25).item()

        # Pad and be careful of boundaries
        x_min = max(x_min - x_padding, 0)
        x_max = min(x_max + x_padding, W-1)
        y_min = max(y_min - y_padding, 0)
        y_max = min(y_max + y_padding, H-1)
        crop_indices[mask_id.item()] = [x_min, y_min, x_max, y_max] # save crop indices

        # Crop
        rgb_crop = rgb_img[:, y_min:y_max+1, x_min:x_max+1] # [3 x crop_H x crop_W]
        mask_crop = mask[y_min:y_max+1, x_min:x_max+1] # [crop_H x crop_W]

        # Resize
        new_size = (224,224)
        rgb_crop = F.upsample_bilinear(rgb_crop.unsqueeze(0), new_size)[0] # Shape: [3 x new_H x new_W]
        rgb_crops[index] = rgb_crop
        mask_crop = F.upsample_nearest(mask_crop.unsqueeze(0).unsqueeze(0), new_size)[0,0] # Shape: [new_H, new_W]
        mask_crops[index] = mask_crop

    # Run the RGB Refinement Network
    if mask_ids.shape[0] > 0: # only run if you actually have masks to refine...

        new_batch = {'rgb' : rgb_crops, 'initial_masks' : mask_crops}
        refined_crops = rgb_refinement_network.run_on_batch(new_batch) # Shape: [num_masks x new_H x new_W]

    # resize the results to the original size. Order this by average depth (highest to lowest)
    sorted_mask_ids = []
    for index, mask_id in enumerate(mask_ids):

        # Resize back to original size
        x_min, y_min, x_max, y_max = crop_indices[mask_id.item()]
        orig_H = y_max - y_min + 1
        orig_W = x_max - x_min + 1
        mask = refined_crops[index].unsqueeze(0).unsqueeze(0).float()
        resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0,0]

        # Calculate average depth
        h_idx, w_idx = torch.nonzero(resized_mask).t()
        avg_depth = torch.mean(xyz_img[2, y_min:y_max+1, x_min:x_max+1][h_idx, w_idx])
        sorted_mask_ids.append((index, mask_id, avg_depth))

    sorted_mask_ids = sorted(sorted_mask_ids, key=lambda x : x[2], reverse=True)
    sorted_mask_ids = [x[:2] for x in sorted_mask_ids] # list of tuples: (index, mask_id)

    for index, mask_id in sorted_mask_ids:

        # Resize back to original size
        x_min, y_min, x_max, y_max = crop_indices[mask_id.item()]
        orig_H = y_max - y_min + 1
        orig_W = x_max - x_min + 1
        mask = refined_crops[index].unsqueeze(0).unsqueeze(0).float()
        resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0,0]

        # Set refined mask
        h_idx, w_idx = torch.nonzero(resized_mask).t()
        refined_masks[y_min:y_max+1, x_min:x_max+1][h_idx, w_idx] = mask_id

                                
    # Open/close morphology stuff, for synthetically-trained RRN
    refined_masks = refined_masks.cpu().numpy() # to CPU

    # Get object ids. Remove background (0)
    obj_ids = np.unique(refined_masks)
    if obj_ids[0] == 0:
        obj_ids = obj_ids[1:]

    # For each object id, open/close the masks
    for obj_id in obj_ids:
        mask = (refined_masks == obj_id) # Shape: [H x W]

        ksize = 9
        opened_mask = cv2.morphologyEx(mask.astype(np.uint8), 
                                       cv2.MORPH_OPEN, 
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))
        opened_closed_mask = cv2.morphologyEx(opened_mask,
                                              cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))

        h_idx, w_idx = np.nonzero(mask)
        refined_masks[h_idx, w_idx] = 0
        h_idx, w_idx = np.nonzero(opened_closed_mask)
        refined_masks[h_idx, w_idx] = obj_id

                                
    # Write out mask to file
    file_path = save_dir + label_abs_path.rsplit('/', 1)[0] + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = file_path + label_abs_path.rsplit('/', 1)[1].rsplit('.', 1)[0] + '.png'
    util_.imwrite_indexed(file_name, refined_masks.astype(np.uint8))
"""

def run_RRN_v5_on_Mask_RCNN_OSD_labels():
    pass
"""
rgb_refinement_network = segmentation.RGBRefinementNetwork(None, rrn_params)
checkpoint_dir = '/home/chrisxie/projects/ssc/checkpoints/'
rgb_refinement_network.load(checkpoint_dir + 'RGBRefinementNetwork_iter100000_TableTop_v5_64c_checkpoint.pth.tar')

-----------------------------------------------------------------------------------------------------------------

import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm
    
H = 480
W = 640
    
osd_filepath = datasets_base_dir + 'OSD/'
starts_file = osd_filepath + 'label_files.txt'
prediction_dir = '/home/chrisxie/projects/ssc/external/OSD_results/Mask_RCNN/RGBD/'
save_dir = '/home/chrisxie/projects/ssc/external/OSD_results/Mask_RCNN/RGBD_RRN_v5/'

# Parse starts file
f = open(starts_file, 'r')
label_filenames = [x.strip() for x in f.readlines()]

# Run the RRN for each thing
for label_filename in tqdm(label_filenames):

    # Load the prediction (reverse the process I used to go from pcd_files.txt to label_files.txt)
    pred_filename = label_filename.replace(osd_filepath, prediction_dir)
    pred_filename = pred_filename.replace('/OSD-0.2-depth/', '/OSD-0.2/')
    pred_filename = pred_filename.replace('/annotation/', '/pcd/')
    pred_img = util_.imread_indexed(pred_filename)
    initial_masks = torch.from_numpy(pred_img).to('cuda') # Shape: [H x W]
    
    ### Load RGB/Depth ###
    pcd_filename = label_filename.replace('-depth', '').replace('annotation', 'pcd').replace('.png', '.pcd')
    print(pcd_filename)

    # Process .pcd file
    temp_idx = pcd_filename.split('/').index('OSD') # parse something like this: /data/OSD/OSD-0.2/pcd/learn44.pcd
    label_abs_path = '/'.join(pcd_filename.split('/')[temp_idx+1:])
    point_cloud = open3d.read_point_cloud(pcd_filename)
    
    # Fill in missing pixel values for RGB
    num_missing = 480*640 - np.asarray(point_cloud.colors).shape[0]
    filled_in_rgb_img = np.concatenate([np.asarray(point_cloud.colors), np.zeros((num_missing,3))])
    rgb_img = np.round(255 * filled_in_rgb_img.reshape(480,640,3)).astype(np.uint8)
    rgb_img = rgb_img.astype(np.float32)
    rgb_img = data_augmentation.BGR_image(rgb_img)
    rgb_img = data_augmentation.array_to_tensor(rgb_img).to('cuda') # Shape: [3 x H x W]
    
    # Fill in missing xyz values
    num_missing = 480*640 - np.asarray(point_cloud.points).shape[0]
    filled_in_points = np.concatenate([np.asarray(point_cloud.points), np.zeros((num_missing,3))])
    xyz_img = np.asarray(filled_in_points).reshape(480,640,3)
    xyz_img[np.isnan(xyz_img)] = 0
    xyz_img = data_augmentation.array_to_tensor(xyz_img).to('cuda') # Shape: [3 x H x W]
    
    ##### Run the RRN #####
    
    # Data structure to hold everything at end
    refined_masks = torch.zeros_like(initial_masks)

    # Dictionary to save crop indices
    crop_indices = {}

    mask_ids = torch.unique(initial_masks)
    if mask_ids[0] == 0: # Get rid of background
        mask_ids = mask_ids[1:]
    if len(mask_ids) > 0 and mask_ids[0] == 1: # Get rid of table
        mask_ids = mask_ids[1:]
    rgb_crops = torch.zeros((mask_ids.shape[0], 3, 224, 224), device='cuda')
    mask_crops = torch.zeros((mask_ids.shape[0], 224, 224), device='cuda')

    for index, mask_id in enumerate(mask_ids):
        mask = (initial_masks == mask_id).float() # Shape: [H x W]

        # crop the masks/rgb to 224x224 with some padding, save it as "initial_masks"
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(mask)
        x_padding = torch.round((x_max - x_min).float() * 0.25).item()
        y_padding = torch.round((y_max - y_min).float() * 0.25).item()

        # Pad and be careful of boundaries
        x_min = max(x_min - x_padding, 0)
        x_max = min(x_max + x_padding, W-1)
        y_min = max(y_min - y_padding, 0)
        y_max = min(y_max + y_padding, H-1)
        crop_indices[mask_id.item()] = [x_min, y_min, x_max, y_max] # save crop indices

        # Crop
        rgb_crop = rgb_img[:, y_min:y_max+1, x_min:x_max+1] # [3 x crop_H x crop_W]
        mask_crop = mask[y_min:y_max+1, x_min:x_max+1] # [crop_H x crop_W]

        # Resize
        new_size = (224,224)
        rgb_crop = F.upsample_bilinear(rgb_crop.unsqueeze(0), new_size)[0] # Shape: [3 x new_H x new_W]
        rgb_crops[index] = rgb_crop
        mask_crop = F.upsample_nearest(mask_crop.unsqueeze(0).unsqueeze(0), new_size)[0,0] # Shape: [new_H, new_W]
        mask_crops[index] = mask_crop

    # Run the RGB Refinement Network
    if mask_ids.shape[0] > 0: # only run if you actually have masks to refine...

        new_batch = {'rgb' : rgb_crops, 'initial_masks' : mask_crops}
        refined_crops = rgb_refinement_network.run_on_batch(new_batch) # Shape: [num_masks x new_H x new_W]

    # resize the results to the original size. Order this by average depth (highest to lowest)
    sorted_mask_ids = []
    for index, mask_id in enumerate(mask_ids):

        # Resize back to original size
        x_min, y_min, x_max, y_max = crop_indices[mask_id.item()]
        orig_H = y_max - y_min + 1
        orig_W = x_max - x_min + 1
        mask = refined_crops[index].unsqueeze(0).unsqueeze(0).float()
        resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0,0]

        # Calculate average depth
        h_idx, w_idx = torch.nonzero(resized_mask).t()
        avg_depth = torch.mean(xyz_img[2, y_min:y_max+1, x_min:x_max+1][h_idx, w_idx])
        sorted_mask_ids.append((index, mask_id, avg_depth))

    sorted_mask_ids = sorted(sorted_mask_ids, key=lambda x : x[2], reverse=True)
    sorted_mask_ids = [x[:2] for x in sorted_mask_ids] # list of tuples: (index, mask_id)

    for index, mask_id in sorted_mask_ids:

        # Resize back to original size
        x_min, y_min, x_max, y_max = crop_indices[mask_id.item()]
        orig_H = y_max - y_min + 1
        orig_W = x_max - x_min + 1
        mask = refined_crops[index].unsqueeze(0).unsqueeze(0).float()
        resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0,0]

        # Set refined mask
        h_idx, w_idx = torch.nonzero(resized_mask).t()
        refined_masks[y_min:y_max+1, x_min:x_max+1][h_idx, w_idx] = mask_id

                                
    # Open/close morphology stuff, for synthetically-trained RRN
    refined_masks = refined_masks.cpu().numpy() # to CPU

    # Get object ids. Remove background (0)
    obj_ids = np.unique(refined_masks)
    if obj_ids[0] == 0:
        obj_ids = obj_ids[1:]

    # For each object id, open/close the masks
    for obj_id in obj_ids:
        mask = (refined_masks == obj_id) # Shape: [H x W]

        ksize = 9
        opened_mask = cv2.morphologyEx(mask.astype(np.uint8), 
                                       cv2.MORPH_OPEN, 
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))
        opened_closed_mask = cv2.morphologyEx(opened_mask,
                                              cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))

        h_idx, w_idx = np.nonzero(mask)
        refined_masks[h_idx, w_idx] = 0
        h_idx, w_idx = np.nonzero(opened_closed_mask)
        refined_masks[h_idx, w_idx] = obj_id

                                
    # Write out mask to file
    file_path = save_dir + label_abs_path.rsplit('/', 1)[0] + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = file_path + label_abs_path.rsplit('/', 1)[1].rsplit('.', 1)[0] + '.png'
    util_.imwrite_indexed(file_name, refined_masks.astype(np.uint8))
"""

def run_LCCP_on_OCID_OSD():
    pass
"""

# Use py2 conda environment

import pypcd
import numpy
import numpy as np
import os, sys
from tqdm import tqdm
from PIL import Image

def imwrite_indexed(filename,array):

    palette_abspath = '/data/tabletop_dataset_v5/palette.txt' # hard-coded filepath
    color_palette = np.loadtxt(palette_abspath, dtype=np.uint8).reshape(-1,3)

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")
    if array.max() > 255:
        raise Exception("Can't save image if it's np.uint8...")

    im = Image.fromarray(array.astype(np.uint8))
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


datasets_base_dir = '/data/'
# data_filepath = datasets_base_dir + 'OCID-dataset/'
# starts_file = data_filepath + 'pcd_files.txt'
data_filepath = datasets_base_dir + 'OSD/'
starts_file = data_filepath + 'pcd_files.txt'

dataset_name = 'OCID' if 'OCID' in data_filepath else 'OSD'

save_dir = '/home/chrisxie/projects/ssc/external/' + ('OSD_results/' if dataset_name == 'OSD' else 'OCID_results/') + 'LCCP/'

# Parse starts file
f = open(starts_file, 'r')
pcd_filenames = [x.strip() for x in f.readlines()]

# LCCP executable
lccp_exe = '/home/chrisxie/local_installations/pcl/build/bin/pcl_example_lccp_segmentation'

for pcd_filename in tqdm(pcd_filenames):
    print(pcd_filename)

    # Run LCCP
    out_filename = '/home/chrisxie/temp/' + dataset_name
    os.system(lccp_exe + ' ' + pcd_filename + ' -o ' + out_filename + ' -v 0.005 -s 0.02 -novis')
    
    # Collect output, process with pypcd
    temp = pypcd.PointCloud.from_path(out_filename + '_out.pcd')
    lccp_img = temp.pc_data['label'].reshape((480,640))
    depth_img = temp.pc_data['z'].reshape((480,640))
    

    ### Post-process the results ###

    # Set depth 0 to background (label of 0)
    depth_0_label = np.unique(lccp_img[depth_img == 0])
    if len(depth_0_label) == 1 and depth_0_label[0] != 0: # switch labels
        lccp_img[lccp_img == 0] = depth_0_label[0]
        lccp_img[depth_img == 0] = 0

    # Set masks smaller than 500 to 0
    mask_labels = np.unique(lccp_img)
    mask_counts = {}
    for mask_id in mask_labels:
        mask_counts[mask_id] = np.count_nonzero(lccp_img == mask_id)
        if mask_counts[mask_id] < 500: # 500, which is min cluster size for graph cuts
            lccp_img[lccp_img == mask_id] = 0
            
    # Set largest non-depth-0 mask to table
    largest_id = sorted(mask_counts.items(), key=lambda x:x[1], reverse=True)[0][0]
    lccp_img[lccp_img == largest_id] = 1

    # Set everything else to be in [2, 3, 4, ...]
    new_mask = np.zeros_like(lccp_img)
    new_mask_ids = np.arange(np.unique(lccp_img).shape[0])
    for i, mask_id in enumerate(np.unique(lccp_img)):
        new_mask[lccp_img == mask_id] = new_mask_ids[i]
    lccp_img = new_mask

    ### Write out the results ###
    if dataset_name == 'OSD':
        temp_idx = pcd_filename.split('/').index('OSD') # parse something like this: /data/OSD/OSD-0.2/pcd/learn44.pcd
    elif dataset_name == 'OCID':
        temp_idx = pcd_filename.split('/').index('OCID-dataset') # parse something like this: /data/OCID-dataset/YCB10/table/top/curved/seq36/pcd/result_2018-08-24-15-13-13.pcd
    label_abs_path = '/'.join(pcd_filename.split('/')[temp_idx+1:])
    
    file_path = save_dir + label_abs_path.rsplit('/', 1)[0] + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = file_path + label_abs_path.rsplit('/', 1)[1].rsplit('.', 1)[0] + '.png'
    imwrite_indexed(file_name, lccp_img)

"""

def write_out_16bit_depth_images_as_blue_red_image():
    pass
"""
import numpy as np
import cv2
import os
import matplotlib
from matplotlib import cm
import sys
sys.path.append('/home/chrisxie/projects/ssc/util/')
import util as util_

cm_jet = matplotlib.cm.get_cmap('jet')

depth_img = '~/temp/depth_00000.png'
temp = cv2.imread(depth_img, cv2.IMREAD_ANYDEPTH).astype(np.float32)
temp = np.round(util_.normalize(temp) * 255).astype(np.uint8)
temp = (cm_jet(temp) * 255).astype(np.uint8)
cv2.imwrite(depth_img, cv2.cvtColor(temp, cv2.COLOR_RGB2BGR))

"""

def evaluation_with_EasyLabel_baselines():
    pass
"""
def evaluate_on_dataset(prediction_dir, gt_dir, dataset_name):

    from tqdm import tqdm_notebook as tqdm # Because i'm using jupyter notebook. This can be something else for command line usage

    if dataset_name == 'OCID' or dataset_name == 'OSD':
        starts_file = gt_dir + 'label_files.txt'
    elif dataset_name == 'RGBO':
        pass # TODO: implement this
    elif dataset_name == 'Synth':
        pass # TODO: implement this. also name it something better than "Synth"...

    # Parse starts file
    f = open(starts_file, 'r')
    label_filenames = [x.strip() for x in f.readlines()]


    metrics = {} # a dictionary of label_filename -> dictionary of metrics
    if dataset_name == 'OCID': # For OCID, keep track of different combos. hard code this. yeah it's ugly
        OCID_subset_metrics = {}

        OCID_subsets = all_combos([['ARID20','ARID10','YCB10'],['table','floor'],['top','bottom']])

        for subset in OCID_subsets:
            OCID_subset_metrics[subset] = {}

    for label_filename in tqdm(label_filenames):

        # Load the GT label file
        if dataset_name == 'OCID':

            temp_idx = label_filename.split('/').index('OCID-dataset') # parse something like this: /data/OCID-dataset/YCB10/table/top/curved/seq36/pcd/result_2018-08-24-15-13-13.pcd
            subset = label_filename.split('/')[temp_idx+1] # one of: [ARID10, ARID20, YCB10]
            supporting_plane = 'floor' if 'floor' in label_filename else 'table'
            height = 'bottom' if 'bottom' in label_filename else 'top'

            label_img = cv2.imread(label_filename, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            if supporting_plane == 'table': # Merge 0 (NaNs) and 1 (non-table/object)
                label_img[label_img == 1] = 0
                label_img = (label_img.astype(int) - 1).clip(min=0).astype(np.uint8)

        elif dataset_name == 'OSD':

            label_img = cv2.imread(label_filename, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            label_img[label_img > 0] = label_img[label_img > 0] + 1 # so values are in [0, 2, 3, ...] (e.g. no table label)

        # Load the prediction (reverse the process I used to go from pcd_files.txt to label_files.txt)
        if dataset_name == 'OCID':
            pred_filename = label_filename.replace(gt_dir, prediction_dir)
            pred_filename = pred_filename.replace('/label/', '/pcd/')
            if 'graphcut' in prediction_dir or \
               'lccp' in prediction_dir or \
               'scenecut' in prediction_dir or \
               'v4r' in prediction_dir:
                if subset == 'ARID10': # these baselines don't have results on ARID10
                    continue
                pred_filename = prediction_dir + subset + '/' + pred_filename.split('/')[-1]
        elif dataset_name == 'OSD':
            pred_filename = label_filename.replace(gt_dir, prediction_dir)
            pred_filename = pred_filename.replace('/OSD-0.2-depth/', '/OSD-0.2/')
            pred_filename = pred_filename.replace('/annotation/', '/pcd/')
        pred_img = util_.imread_indexed(pred_filename)


        if 'graphcut' in prediction_dir or \
           'lccp' in prediction_dir or \
           'scenecut' in prediction_dir or \
           'v4r' in prediction_dir:

            import open3d

            # Load in depth image
            pcd_filename = label_filename.replace('/label/', '/pcd/').replace('.png', '.pcd')
            point_cloud = open3d.read_point_cloud(pcd_filename)
            num_missing = 480*640 - np.asarray(point_cloud.points).shape[0]
            filled_in_points = np.concatenate([np.asarray(point_cloud.points), np.zeros((num_missing,3))])
            xyz_img = np.asarray(filled_in_points).reshape(480,640,3)
            xyz_img[np.isnan(xyz_img)] = 0
            depth_img = xyz_img[..., 2]

            # Set depth 0 to background (label of 0)
            depth_0_label = np.unique(pred_img[depth_img == 0])
            if len(depth_0_label) == 1 and depth_0_label[0] != 0: # switch labels
                pred_img[pred_img == 0] = depth_0_label[0]
                pred_img[depth_img == 0] = 0

            # Set masks smaller than 500 to 0
            mask_labels = np.unique(pred_img)
            mask_counts = {}
            for mask_id in mask_labels:
                mask_counts[mask_id] = np.count_nonzero(pred_img == mask_id)
                if mask_counts[mask_id] < 500: # 500, which is min cluster size for graph cuts
                    pred_img[pred_img == mask_id] = 0
                    
            # Set largest non-depth-0 mask to table
            largest_id = sorted(mask_counts.items(), key=lambda x:x[1], reverse=True)[0][0]
            pred_img[pred_img == largest_id] = 1

            # Set everything else to be in [2, 3, 4, ...]
            new_mask = np.zeros_like(pred_img)
            new_mask_ids = np.arange(np.unique(pred_img).shape[0])
            for i, mask_id in enumerate(np.unique(pred_img)):
                new_mask[pred_img == mask_id] = new_mask_ids[i]
            pred_img = new_mask



        # Compare them
        metrics_dict = multilabel_metrics(pred_img, label_img)
        metrics[label_filename] = metrics_dict

        if dataset_name == 'OCID':
            OCID_subset_metrics[subset][label_filename] = metrics_dict
            OCID_subset_metrics[subset + '_' + supporting_plane][label_filename] = metrics_dict
            OCID_subset_metrics[subset + '_' + supporting_plane + '_' + height][label_filename] = metrics_dict

        # Debugging
        # print(label_filename)
        # print(f'Overlap  F: {metrics_dict["Objects F-measure"]}')
        # print(f'Boundary F: {metrics_dict["Boundary F-measure"]}')

    # Compute mean of all metrics
    obj_F_mean = np.mean([metrics[key]['Objects F-measure'] for key in metrics.keys()])
    obj_P_mean = np.mean([metrics[key]['Objects Precision'] for key in metrics.keys()])
    obj_R_mean = np.mean([metrics[key]['Objects Recall'] for key in metrics.keys()])
    boundary_F_mean = np.mean([metrics[key]['Boundary F-measure'] for key in metrics.keys()])
    boundary_P_mean = np.mean([metrics[key]['Boundary Precision'] for key in metrics.keys()])
    boundary_R_mean = np.mean([metrics[key]['Boundary Recall'] for key in metrics.keys()])
    obj_det_075_percentage_mean = np.mean([metrics[key]['obj_detected_075_percentage'] for key in metrics.keys()])

    ret_dict = {
        'obj_F_mean' : obj_F_mean,
        'obj_P_mean' : obj_P_mean,
        'obj_R_mean' : obj_R_mean,
        'boundary_F_mean' : boundary_F_mean,
        'boundary_P_mean' : boundary_P_mean,
        'boundary_R_mean' : boundary_R_mean,
        'obj_det_075_percentage_mean' : obj_det_075_percentage_mean,
    }


    if dataset_name == 'OCID':

        # Get every subset
        for subset in OCID_subsets:

            mdict = OCID_subset_metrics[subset]
            obj_F_mean = np.mean([mdict[key]['Objects F-measure'] for key in mdict.keys()])
            obj_P_mean = np.mean([mdict[key]['Objects Precision'] for key in mdict.keys()])
            obj_R_mean = np.mean([mdict[key]['Objects Recall'] for key in mdict.keys()])
            boundary_F_mean = np.mean([mdict[key]['Boundary F-measure'] for key in mdict.keys()])
            boundary_P_mean = np.mean([mdict[key]['Boundary Precision'] for key in mdict.keys()])
            boundary_R_mean = np.mean([mdict[key]['Boundary Recall'] for key in mdict.keys()])
            obj_det_075_percentage_mean = np.mean([mdict[key]['obj_detected_075_percentage'] for key in mdict.keys()])

            ret_dict[subset] = {
                'obj_F_mean' : obj_F_mean,
                'obj_P_mean' : obj_P_mean,
                'obj_R_mean' : obj_R_mean,
                'boundary_F_mean' : boundary_F_mean,
                'boundary_P_mean' : boundary_P_mean,
                'boundary_R_mean' : boundary_R_mean,
                'obj_det_075_percentage_mean' : obj_det_075_percentage_mean,
            }

    return ret_dict
"""

def visualize_segmentation_results_for_paper():
    pass
"""
# RGB, Depth, Foreground, Center Directions, Initial Masks (after IMP), Refined Mask

data_loader = reload(data_loader)

# ocid_filepath = datasets_base_dir + 'OCID-dataset/'
# dl = data_loader.get_OCID_dataloader(ocid_filepath, batch_size=10, num_workers=8, shuffle=True)

# osd_filepath = datasets_base_dir + 'OSD/'
# dl = data_loader.get_OSD_dataloader(osd_filepath, batch_size=10, num_workers=8, shuffle=False)

real_rgbd_images_filepath = datasets_base_dir + 'tabletop_dataset_v5/real_RGBD_images/'
dl = data_loader.get_Real_RGBD_Images_dataloader(real_rgbd_images_filepath, {},
                                                 batch_size=10, num_workers=8, shuffle=False)

dl_iter = dl.__iter__()

---------------------------------------------------------------------------------------------------

batch = next(dl_iter)
rgb_imgs = torch_to_numpy(batch['rgb'], is_standardized_image=True) # Shape: [N x H x W x 3]
xyz_imgs = torch_to_numpy(batch['xyz']) # Shape: [N x H x W x 3]
N, H, W = rgb_imgs.shape[:3]

---------------------------------------------------------------------------------------------------

print("Number of images: {0}".format(N))

### Compute segmentation masks ###
st_time = time()
fg_masks, direction_predictions, initial_masks, plane_masks, distance_from_table, seg_masks = tabletop_segmentor.run_on_batch(batch)
total_time = time() - st_time
print('Total time taken for Segmentation: {0} seconds'.format(round(total_time, 3)))
print('FPS: {0}'.format(round(N / total_time,3)))

# Get segmentation masks in numpy
seg_masks = seg_masks.cpu().numpy()

### Debug stuff
fg_masks = fg_masks.cpu().numpy()
direction_predictions = direction_predictions.cpu().numpy().transpose(0,2,3,1)
initial_masks = initial_masks.cpu().numpy()
plane_masks = plane_masks.cpu().numpy()
distance_from_table = distance_from_table.cpu().numpy()

---------------------------------------------------------------------------------------------------

rgb_imgs = torch_to_numpy(batch['rgb'].cpu(), is_standardized_image=True)
total_subplots = 6

fig_index = 1
for i in range(N):
    
    fig = plt.figure(fig_index); fig_index += 1
    fig.set_size_inches(20,5)

    # Plot image
    plt.subplot(1,total_subplots,1)
    plt.imshow(rgb_imgs[i,...].astype(np.uint8))
#     plt.title(f"{batch['label_abs_path'][i].rsplit('/',1)[-1]}")

    # Plot Depth
    plt.subplot(1,total_subplots,2)
    plt.imshow(xyz_imgs[i,...,2])
    plt.title('Depth')
    
    # Plot initial masks
    plt.subplot(1,total_subplots,3)
    plt.imshow(util_.get_color_mask(initial_masks[i,...]))
    plt.title(f"Initial Masks. #objects: {np.unique(initial_masks[i,...]).shape[0]-1}")

    # Plot direction predictions
    plt.subplot(1,total_subplots,4)
    plt.imshow(flowlib.flow_to_image(direction_predictions[i,...]))
    plt.title("Center Direction Predictions")
    
    # Plot Masks
    plt.subplot(1,total_subplots,5)
    plt.imshow(util_.get_color_mask(seg_masks[i,...]))
    plt.title(f"Refined Masks. #objects: {np.unique(seg_masks[i,...]).shape[0]-1}")
    
    # Plot foreground mask
    plt.subplot(1,total_subplots,6)
    plt.imshow(util_.get_color_mask(fg_masks[i,...]))
    plt.title("Foreground Mask")

---------------------------------------------------------------------------------------------------

import matplotlib
from matplotlib import cm
from PIL import Image

i = 1
save_dir = '/home/chrisxie/temp/'
os.system('rm -f ' + save_dir + '/*')

# image
rgb_img = rgb_imgs[i,...].astype(np.uint8)
im = Image.fromarray(rgb_img)
im.save(save_dir + 'rgb.png', format='PNG')

# Depth
cm_jet = matplotlib.cm.get_cmap('jet')
depth_img = np.round(xyz_imgs[i,...,2] / xyz_imgs[i,...,2].max() * 255).astype(np.uint8)
depth_img = (cm_jet(depth_img) * 255).astype(np.uint8)
im = Image.fromarray(depth_img)
im.save(save_dir + 'depth.png', format='PNG')

# initial masks
initial_mask = util_.get_color_mask(initial_masks[i,...])
im = Image.fromarray(initial_mask)
im.save(save_dir + 'initial_masks_after_imp.png', format='PNG')
visualized_initial_mask = util_.visualize_segmentation(rgb_img, initial_masks[i,...], nc=None, return_rgb=True, save_dir=None)
im = Image.fromarray(visualized_initial_mask)
im.save(save_dir + 'visualized_initial_masks_after_imp.png', format='PNG')

# direction predictions
im = Image.fromarray(flowlib.flow_to_image(direction_predictions[i,...]))
im.save(save_dir + 'center_direction_prediction.png', format='PNG')

# Refined Masks
refined_mask = util_.get_color_mask(seg_masks[i,...])
im = Image.fromarray(refined_mask)
im.save(save_dir + 'refined_masks.png', format='PNG')
visualized_refined_mask = util_.visualize_segmentation(rgb_img, seg_masks[i,...], nc=None, return_rgb=True, save_dir=None)
im = Image.fromarray(visualized_refined_mask)
im.save(save_dir + 'visualized_refined_masks.png', format='PNG')

# foreground mask
im = Image.fromarray(util_.get_color_mask(fg_masks[i,...]))
im.save(save_dir + 'foreground_masks.png', format='PNG')


"""

def visualize_ttsnet_vs_maskrcnn():
    pass
"""
# ttsnet_results_dir = '/home/chrisxie/projects/ssc/external/OCID_results/TTSNet_DSN_v5_RRN_v5_IMP_oc_ccc/'
# maskrcnn_results_dir = '/home/chrisxie/projects/ssc/external/OCID_results/Mask_RCNN/RGBD/'
# data_filepath = datasets_base_dir + 'OCID-dataset/'
# starts_file = data_filepath + 'label_files.txt'

ttsnet_results_dir = '/home/chrisxie/projects/ssc/external/OSD_results/TTSNet_DSN_v5_RRN_v5_IMP_oc_ccc/'
maskrcnn_results_dir = '/home/chrisxie/projects/ssc/external/OSD_results/Mask_RCNN/RGBD/'
data_filepath = datasets_base_dir + 'OSD/'
starts_file = data_filepath + 'label_files.txt'

dataset_name = 'OCID' if 'OCID' in data_filepath else 'OSD'

# Parse starts file
f = open(starts_file, 'r')
label_filenames = [x.strip() for x in f.readlines()]

# random label selection
label_filenames = np.random.permutation(label_filenames)[:4]

ttsnet_images = []
maskrcnn_images = []

for i, label_filename in enumerate(label_filenames):
    
    # Load the ttsnet predictions
    pred_filename = label_filename.replace(data_filepath, ttsnet_results_dir)
    if dataset_name == 'OCID':
        pred_filename = pred_filename.replace('/label/', '/pcd/')
    elif dataset_name == 'OSD':
        pred_filename = pred_filename.replace('/OSD-0.2-depth/', '/OSD-0.2/')
        pred_filename = pred_filename.replace('/annotation/', '/pcd/')
    ttsnet_pred_img = util_.imread_indexed(pred_filename)
    
    # Load the Mask RCNN predictions
    pred_filename = label_filename.replace(data_filepath, maskrcnn_results_dir)
    if dataset_name == 'OCID':
        pred_filename = pred_filename.replace('/label/', '/pcd/')
    elif dataset_name == 'OSD':
        pred_filename = pred_filename.replace('/OSD-0.2-depth/', '/OSD-0.2/')
        pred_filename = pred_filename.replace('/annotation/', '/pcd/')
    maskrcnn_pred_img = util_.imread_indexed(pred_filename)
    maskrcnn_pred_img[maskrcnn_pred_img == 1] = 0 # turn table into background
    
    # Load RGB
    if dataset_name == 'OCID':
        pcd_filename = label_filename.replace('/label/', '/pcd/').replace('.png', '.pcd')
    elif dataset_name == 'OSD':
        pcd_filename = label_filename.replace('-depth', '').replace('annotation', 'pcd').replace('.png', '.pcd')
    point_cloud = open3d.read_point_cloud(pcd_filename)
    # Fill in missing pixel values for RGB
    num_missing = 480*640 - np.asarray(point_cloud.colors).shape[0] 
    filled_in_rgb_img = np.concatenate([np.asarray(point_cloud.colors), np.zeros((num_missing,3))])
    rgb_img = np.round(255 * filled_in_rgb_img.reshape(480,640,3)).astype(np.uint8)
    
    # Visualize the segmentations
    ttsnet_pred_img = util_.visualize_segmentation(rgb_img, ttsnet_pred_img, nc=None, return_rgb=True, save_dir=None)
    maskrcnn_pred_img = util_.visualize_segmentation(rgb_img, maskrcnn_pred_img, nc=None, return_rgb=True, save_dir=None)
    
    ttsnet_images.append(ttsnet_pred_img)
    maskrcnn_images.append(maskrcnn_pred_img)
    
fig = plt.figure(1); fig.set_size_inches(20,5)
for i in range(len(ttsnet_images)):
    plt.subplot(1,len(ttsnet_images),i+1)
    plt.imshow(ttsnet_images[i])
    plt.title(label_filenames[i])

fig = plt.figure(2); fig.set_size_inches(20,5)
for i in range(len(maskrcnn_images)):
    plt.subplot(1,len(maskrcnn_images),i+1)
    plt.imshow(maskrcnn_images[i])
    plt.title(label_filenames[i])

---------------------------------------------------------------------------------------------------

from PIL import Image
save_dir = '/home/chrisxie/temp/'
for i in range(len(ttsnet_images)):
    
    im = Image.fromarray(ttsnet_images[i])
    im_name = label_filenames[i].split('/')[-1]
    im.save(save_dir + f'ttsnet_{im_name}', format='PNG')

"""

def visualize_graphcuts_baseline():
    pass
"""
import skimage.segmentation
from tqdm import tqdm_notebook as tqdm # Because i'm using jupyter notebook. This can be something else for command line usage

# data_filepath = datasets_base_dir + 'OCID-dataset/'
# starts_file = data_filepath + 'pcd_files.txt'
data_filepath = datasets_base_dir + 'OSD/'
starts_file = data_filepath + 'pcd_files.txt'

dataset_name = 'OCID' if 'OCID' in data_filepath else 'OSD'

save_dir = '/home/chrisxie/projects/ssc/external/' + ('OSD_results/' if dataset_name == 'OSD' else 'OCID_results/') + 'GCUT/'

# Parse starts file
f = open(starts_file, 'r')
pcd_filenames = [x.strip() for x in f.readlines()]

for i, pcd_filename in enumerate(tqdm(pcd_filenames)):
    print(pcd_filename)
    
    # Load RGB
    point_cloud = open3d.read_point_cloud(pcd_filename)
    # Fill in missing pixel values for RGB
    num_missing = 480*640 - np.asarray(point_cloud.colors).shape[0] 
    filled_in_rgb_img = np.concatenate([np.asarray(point_cloud.colors), np.zeros((num_missing,3))])
    rgb_img = np.round(255 * filled_in_rgb_img.reshape(480,640,3)).astype(np.uint8)

    # Run Graph Cuts with parameters specified in OCID paper
    gcut_img = skimage.segmentation.felzenszwalb(rgb_img, scale=500, sigma=0.4, min_size=500)

    ### Write out the results ###
    
    if dataset_name == 'OSD':
        temp_idx = pcd_filename.split('/').index('OSD') # parse something like this: /data/OSD/OSD-0.2/pcd/learn44.pcd
    elif dataset_name == 'OCID':
        temp_idx = pcd_filename.split('/').index('OCID-dataset') # parse something like this: /data/OCID-dataset/YCB10/table/top/curved/seq36/pcd/result_2018-08-24-15-13-13.pcd
    label_abs_path = '/'.join(pcd_filename.split('/')[temp_idx+1:])
    
    file_path = save_dir + label_abs_path.rsplit('/', 1)[0] + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = file_path + label_abs_path.rsplit('/', 1)[1].rsplit('.', 1)[0] + '.png'
    util_.imwrite_indexed(file_name, gcut_img.astype(np.uint8))
"""

def visualize_lccp_results():
    pass
"""
from PIL import Image

# results_dir = '/home/chrisxie/projects/ssc/external/OCID_results/LCCP/'
# data_filepath = datasets_base_dir + 'OCID-dataset/'
# starts_file = data_filepath + 'label_files.txt'
results_dir = '/home/chrisxie/projects/ssc/external/OSD_results/LCCP/'
data_filepath = datasets_base_dir + 'OSD/'
starts_file = data_filepath + 'label_files.txt'

dataset_name = 'OCID' if 'OCID' in data_filepath else 'OSD'

# Parse starts file
f = open(starts_file, 'r')
label_filenames = [x.strip() for x in f.readlines()]

# Create two images
final_image = np.ones((480, 640 * 4 + 50*3, 3), dtype=np.uint8) * 255

# random label selection
label_filenames = np.random.permutation(label_filenames)[:4]

for i, label_filename in enumerate(label_filenames):
    
    # Load the LCCP predictions
    pred_filename = label_filename.replace(data_filepath, results_dir)
    if dataset_name == 'OCID':
        pred_filename = pred_filename.replace('/label/', '/pcd/')
    elif dataset_name == 'OSD':
        pred_filename = pred_filename.replace('/OSD-0.2-depth/', '/OSD-0.2/')
        pred_filename = pred_filename.replace('/annotation/', '/pcd/')
    pred_img = util_.imread_indexed(pred_filename)
    
    # Load RGB
    if dataset_name == 'OCID':
        pcd_filename = label_filename.replace('/label/', '/pcd/').replace('.png', '.pcd')
    elif dataset_name == 'OSD':
        pcd_filename = label_filename.replace('-depth', '').replace('annotation', 'pcd').replace('.png', '.pcd')
    point_cloud = open3d.read_point_cloud(pcd_filename)
    # Fill in missing pixel values for RGB
    num_missing = 480*640 - np.asarray(point_cloud.colors).shape[0] 
    filled_in_rgb_img = np.concatenate([np.asarray(point_cloud.colors), np.zeros((num_missing,3))])
    rgb_img = np.round(255 * filled_in_rgb_img.reshape(480,640,3)).astype(np.uint8)
    
    # Visualize the segmentations
    pred_img = util_.visualize_segmentation(rgb_img, pred_img, nc=None, return_rgb=True, save_dir=None)
    
    # Write them
    final_image[:, i*(640+50):((i+1)*640 + i*50), :] = pred_img
    
# im = Image.fromarray(final_image)
# im.save('/home/chrisxie/temp/lccp_image.png', format='PNG')
    
plt.figure(1, figsize=(20,5))
plt.imshow(final_image)

"""

def visualize_my_results_against_4_easylabel_baselines_and_maskrcnn_OCID():
    pass
"""
### Visualize my results against 4 EasyLabel baselines and MaskRCNN on OCID

def load_easylabel_baseline_img(label_filename, pred_filename):
    pred_img = util_.imread_indexed(pred_filename)
    
    import open3d

    # Load in depth image
    pcd_filename = label_filename.replace('/label/', '/pcd/').replace('.png', '.pcd')
    point_cloud = open3d.read_point_cloud(pcd_filename)
    num_missing = 480*640 - np.asarray(point_cloud.points).shape[0]
    filled_in_points = np.concatenate([np.asarray(point_cloud.points), np.zeros((num_missing,3))])
    xyz_img = np.asarray(filled_in_points).reshape(480,640,3)
    xyz_img[np.isnan(xyz_img)] = 0
    depth_img = xyz_img[..., 2]

    # Set depth 0 to background (label of 0)
    depth_0_label = np.unique(pred_img[depth_img == 0])
    if len(depth_0_label) == 1 and depth_0_label[0] != 0: # switch labels
        pred_img[pred_img == 0] = depth_0_label[0]
        pred_img[depth_img == 0] = 0

    # Set masks smaller than 500 to 0
    mask_labels = np.unique(pred_img)
    mask_counts = {}
    for mask_id in mask_labels:
        mask_counts[mask_id] = np.count_nonzero(pred_img == mask_id)
        if mask_counts[mask_id] < 500: # 500, which is min cluster size for graph cuts
            pred_img[pred_img == mask_id] = 0

    # Set largest non-depth-0 mask to table
    largest_id = sorted(mask_counts.items(), key=lambda x:x[1], reverse=True)[0][0]
    pred_img[pred_img == largest_id] = 0

    # Set everything else to be in [2, 3, 4, ...]
    new_mask = np.zeros_like(pred_img)
    new_mask_ids = np.arange(np.unique(pred_img).shape[0])
    for i, mask_id in enumerate(np.unique(pred_img)):
        new_mask[pred_img == mask_id] = new_mask_ids[i]
    pred_img = new_mask
    
    return pred_img

---------------------------------------------------------------------------------------------------

%matplotlib inline
ttsnet_results_dir = '/home/chrisxie/projects/ssc/external/OCID_results/TTSNet_DSN_v5_RRN_v5_IMP_oc_ccc/'
maskrcnn_results_dir = '/home/chrisxie/projects/ssc/external/OCID_results/Mask_RCNN/RGBD/'
gcut_results_dir = '/home/chrisxie/Downloads/OCID_baseline_results/graphcut/'
scut_results_dir = '/home/chrisxie/Downloads/OCID_baseline_results/scenecut/'
lccp_results_dir = '/home/chrisxie/Downloads/OCID_baseline_results/lccp/'
v4r_results_dir = '/home/chrisxie/Downloads/OCID_baseline_results/v4r/'
data_filepath = datasets_base_dir + 'OCID-dataset/'
starts_file = data_filepath + 'label_files.txt'

dataset_name = 'OCID'

# Parse starts file
f = open(starts_file, 'r')
label_filenames = [x.strip() for x in f.readlines()]

# random label selection
label_filenames = [x for x in label_filenames if 'ARID20' in x or 'YCB10' in x]
label_filename = np.random.permutation(label_filenames)[0]


# Load the ttsnet predictions
pred_filename = label_filename.replace(data_filepath, ttsnet_results_dir)
pred_filename = pred_filename.replace('/label/', '/pcd/')
ttsnet_pred_img = util_.imread_indexed(pred_filename)

# Load the Mask RCNN predictions
pred_filename = label_filename.replace(data_filepath, maskrcnn_results_dir)
pred_filename = pred_filename.replace('/label/', '/pcd/')
maskrcnn_pred_img = util_.imread_indexed(pred_filename)
maskrcnn_pred_img[maskrcnn_pred_img == 1] = 0 # turn table into background

temp_idx = label_filename.split('/').index('OCID-dataset') # parse something like this: /data/OCID-dataset/YCB10/table/top/curved/seq36/pcd/result_2018-08-24-15-13-13.pcd
subset = label_filename.split('/')[temp_idx+1] # one of: [ARID10, ARID20, YCB10]

# Load GCUT prediction
pred_filename = gcut_results_dir + subset + '/' + pred_filename.split('/')[-1]
gcut_pred_img = load_easylabel_baseline_img(label_filename, pred_filename)

# Load SCUT prediction
pred_filename = scut_results_dir + subset + '/' + pred_filename.split('/')[-1]
scut_pred_img = load_easylabel_baseline_img(label_filename, pred_filename)

# Load LCCP prediction
pred_filename = lccp_results_dir + subset + '/' + pred_filename.split('/')[-1]
lccp_pred_img = load_easylabel_baseline_img(label_filename, pred_filename)

# Load V4R prediction
pred_filename = v4r_results_dir + subset + '/' + pred_filename.split('/')[-1]
v4r_pred_img = load_easylabel_baseline_img(label_filename, pred_filename)



# Load RGB
pcd_filename = label_filename.replace('/label/', '/pcd/').replace('.png', '.pcd')
point_cloud = open3d.read_point_cloud(pcd_filename)
# Fill in missing pixel values for RGB
num_missing = 480*640 - np.asarray(point_cloud.colors).shape[0] 
filled_in_rgb_img = np.concatenate([np.asarray(point_cloud.colors), np.zeros((num_missing,3))])
rgb_img = np.round(255 * filled_in_rgb_img.reshape(480,640,3)).astype(np.uint8)

# Visualize the segmentations
ttsnet_pred_img = util_.visualize_segmentation(rgb_img, ttsnet_pred_img, nc=None, return_rgb=True, save_dir=None)
maskrcnn_pred_img = util_.visualize_segmentation(rgb_img, maskrcnn_pred_img, nc=None, return_rgb=True, save_dir=None)
gcut_pred_img = util_.visualize_segmentation(rgb_img, gcut_pred_img, nc=None, return_rgb=True, save_dir=None)
scut_pred_img = util_.visualize_segmentation(rgb_img, scut_pred_img, nc=None, return_rgb=True, save_dir=None)
lccp_pred_img = util_.visualize_segmentation(rgb_img, lccp_pred_img, nc=None, return_rgb=True, save_dir=None)
v4r_pred_img = util_.visualize_segmentation(rgb_img, v4r_pred_img, nc=None, return_rgb=True, save_dir=None)
    
    
    
# Plotting
fig = plt.figure(1); 
fig.set_size_inches(15,5)

plt.subplot(1,3,1)
plt.imshow(ttsnet_pred_img)
plt.title('TTS-Net')

plt.subplot(1,3,2)
plt.imshow(maskrcnn_pred_img)
plt.title('Mask RCNN')

plt.subplot(1,3,3)
plt.imshow(gcut_pred_img)
plt.title('GCUT')

fig = plt.figure(2); 
fig.set_size_inches(15,5)

plt.subplot(1,3,1)
plt.imshow(scut_pred_img)
plt.title('SCUT')

plt.subplot(1,3,2)
plt.imshow(lccp_pred_img)
plt.title('LCCP')

plt.subplot(1,3,3)
plt.imshow(v4r_pred_img)
plt.title('V4R')

---------------------------------------------------------------------------------------------------
from PIL import Image

# Save images
save_dir = '/home/chrisxie/temp/'
os.system('rm -f ' + save_dir + '/*')

ttsnet_pred_img = ttsnet_pred_img.astype(np.uint8)
im = Image.fromarray(ttsnet_pred_img)
im.save(save_dir + 'ttsnet.png', format='PNG')

maskrcnn_pred_img = maskrcnn_pred_img.astype(np.uint8)
im = Image.fromarray(maskrcnn_pred_img)
im.save(save_dir + 'maskrcnn.png', format='PNG')

gcut_pred_img = gcut_pred_img.astype(np.uint8)
im = Image.fromarray(gcut_pred_img)
im.save(save_dir + 'gcut.png', format='PNG')

scut_pred_img = scut_pred_img.astype(np.uint8)
im = Image.fromarray(scut_pred_img)
im.save(save_dir + 'scut.png', format='PNG')

lccp_pred_img = lccp_pred_img.astype(np.uint8)
im = Image.fromarray(lccp_pred_img)
im.save(save_dir + 'lccp.png', format='PNG')

v4r_pred_img = v4r_pred_img.astype(np.uint8)
im = Image.fromarray(v4r_pred_img)
im.save(save_dir + 'V4R.png', format='PNG')

"""

def generate_sequences_of_LCCP_V4R_MaskRCNN_Ours():
    pass
"""
from PIL import Image
import matplotlib
from matplotlib import cm
cm_jet = matplotlib.cm.get_cmap('jet')


ttsnet_results_dir = '/home/chrisxie/projects/ssc/external/OCID_results/TTSNet_DSN_v5_RRN_v5_IMP_oc_ccc/'
maskrcnn_results_dir = '/home/chrisxie/projects/ssc/external/OCID_results/Mask_RCNN/RGBD/'
lccp_results_dir = '/home/chrisxie/Downloads/OCID_baseline_results/lccp/'
v4r_results_dir = '/home/chrisxie/Downloads/OCID_baseline_results/v4r/'
data_filepath = datasets_base_dir + 'OCID-dataset/'
starts_file = data_filepath + 'label_files.txt'

dataset_name = 'OCID'

save_dir = '/home/chrisxie/temp/'
os.system(f'rm -f {save_dir}*')

# Parse starts file
f = open(starts_file, 'r')
label_filenames = [x.strip() for x in f.readlines()]

# random sequece selection
label_filenames = [x for x in label_filenames if 'YCB10' in x] # switch between ARID20 and YCB10
label_filename = np.random.permutation(label_filenames)[0]
subset = label_filename.split('/')[3]
seq = '/'.join(label_filename.split('/')[4:-1])
label_filenames = sorted(glob.glob(data_filepath + subset + '/' + seq + '/*.png'))
N = len(label_filenames)

pix_btwn_tb = 100
pix_btwn_lr = 100
images = np.ones((N, 480*2 + pix_btwn_tb*2, 640*3 + 2*pix_btwn_lr, 3), dtype=np.uint8) * 255

for i, label_filename in enumerate(label_filenames):

    ### Load RGB and Depth ###
    pcd_filename = label_filename.replace('/label/', '/pcd/').replace('.png', '.pcd')
    print(pcd_filename)

    # Process .pcd file 
    temp_idx = pcd_filename.split('/').index('OCID-dataset') # parse something like this: /data/OCID-dataset/YCB10/table/top/curved/seq36/pcd/result_2018-08-24-15-13-13.pcd
    label_abs_path = '/'.join(pcd_filename.split('/')[temp_idx+1:])
    point_cloud = open3d.read_point_cloud(pcd_filename)
    
    # Fill in missing pixel values for RGB
    num_missing = 480*640 - np.asarray(point_cloud.colors).shape[0]
    filled_in_rgb_img = np.concatenate([np.asarray(point_cloud.colors), np.zeros((num_missing,3))])
    rgb_img = np.round(255 * filled_in_rgb_img.reshape(480,640,3)).astype(np.uint8)
    
    # Fill in missing xyz values
    num_missing = 480*640 - np.asarray(point_cloud.points).shape[0]
    filled_in_points = np.concatenate([np.asarray(point_cloud.points), np.zeros((num_missing,3))])
    xyz_img = np.asarray(filled_in_points).reshape(480,640,3)
    xyz_img[np.isnan(xyz_img)] = 0
    depth_img = xyz_img[..., 2]
    depth_img = np.round(util_.normalize(depth_img) * 255).astype(np.uint8)
    depth_img = (cm_jet(depth_img) * 255).astype(np.uint8)[..., :3]
    
    
    ### Lay GT on top of RGB ###
    label_img = cv2.imread(label_filename, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    supporting_plane = 'floor' if 'floor' in pcd_filename else 'table'
    if supporting_plane == 'table': # Merge 0 (NaNs) and 1 (non-table/object)
        label_img[label_img == 1] = 0
        label_img = (label_img.astype(int) - 1).clip(min=0).astype(np.uint8)
    label_img[label_img == 1] = 0 # don't show table in visualization
    label_img = util_.visualize_segmentation(rgb_img, label_img, nc=None, return_rgb=True, save_dir=None)

    ### Write RGB+GT and Depth ###
    images[i, 0:480, 0:640, :] = label_img
    images[i, 480+pix_btwn_tb:480*2+pix_btwn_lr, 0:640, :] = depth_img
    
    
    ### Load SOTA predictions ###

    # Load the Mask RCNN predictions
    pred_filename = label_filename.replace(data_filepath, maskrcnn_results_dir)
    pred_filename = pred_filename.replace('/label/', '/pcd/')
    maskrcnn_pred_img = util_.imread_indexed(pred_filename)
    maskrcnn_pred_img[maskrcnn_pred_img == 1] = 0 # turn table into background
    maskrcnn_pred_img = util_.visualize_segmentation(rgb_img, maskrcnn_pred_img, nc=None, return_rgb=True, save_dir=None)
    
    # Load the ttsnet predictions
    pred_filename = label_filename.replace(data_filepath, ttsnet_results_dir)
    pred_filename = pred_filename.replace('/label/', '/pcd/')
    ttsnet_pred_img = util_.imread_indexed(pred_filename)
    ttsnet_pred_img = util_.visualize_segmentation(rgb_img, ttsnet_pred_img, nc=None, return_rgb=True, save_dir=None)

    # Write them in
    images[i, 0:480, 640*2+pix_btwn_lr*2:640*3+pix_btwn_lr*2, :] = maskrcnn_pred_img
    images[i, 480+pix_btwn_tb:480*2+pix_btwn_lr, 640*2+pix_btwn_lr*2:640*3+pix_btwn_lr*2, :] = ttsnet_pred_img
    
    
    ### Load baseline predictions ###
    # Load LCCP prediction
    pred_filename = lccp_results_dir + subset + '/' + pred_filename.split('/')[-1]
    lccp_pred_img = load_easylabel_baseline_img(label_filename, pred_filename)
    lccp_pred_img = util_.visualize_segmentation(rgb_img, lccp_pred_img, nc=None, return_rgb=True, save_dir=None)

    # Load V4R prediction
    pred_filename = v4r_results_dir + subset + '/' + pred_filename.split('/')[-1]
    v4r_pred_img = load_easylabel_baseline_img(label_filename, pred_filename)
    v4r_pred_img = util_.visualize_segmentation(rgb_img, v4r_pred_img, nc=None, return_rgb=True, save_dir=None)

    # Write them in
    images[i, 0:480, 640+pix_btwn_lr:640*2+pix_btwn_lr, :] = lccp_pred_img
    images[i, 480+pix_btwn_tb:480*2+pix_btwn_lr, 640+pix_btwn_lr:640*2+pix_btwn_lr, :] = v4r_pred_img

    
    im = Image.fromarray(images[i])
    im.save(save_dir + f"{subset}_{seq.replace('/','_')}_{i+1}.png", format='PNG')
    
"""

def save_photos_as_npy_files():
    pass
""" Save photos as .npy files for code release CoRL 2019

# Photos to save: Arsalan's photos, lab photos (from Intel RealSense D415), OSD, OCID




### Arsalan's images ###

example_images_dir = '/home/chrisxie/projects/ssc/example_images/'
image_files = sorted(glob.glob(example_images_dir + '/*.npy'))

for i, img_file in enumerate(image_files):
    d = np.load(img_file, allow_pickle=True, encoding='bytes').item()
    rgb_img = d[b'rgb'][..., ::-1] # BGR to RGB image
    depth_img = d[b'depth']
    
    np.save(img_file.replace('_back', ''), {'rgb': rgb_img, 'depth': depth_img})




### Lab Photos ###

rgb_images = sorted(glob.glob('/data/tabletop_dataset_v5/real_RGBD_images/rgb*'))
depth_images = sorted(glob.glob('/data/tabletop_dataset_v5/real_RGBD_images/depth*'))

N = len(depth_images)

save_dir = '/home/chrisxie/projects/ssc/example_images/lab_photos/'

for i in range(N):

    rgb_img = cv2.cvtColor(cv2.imread(rgb_images[i]), cv2.COLOR_BGR2RGB)
    depth_img = cv2.imread(depth_images[i], cv2.IMREAD_ANYDEPTH)

    np.save(f'{save_dir}lab_photo_{i}.npy', {'rgb': rgb_img, 'depth': depth_img})




### OSD ###

f = open('/data/OSD/pcd_files.txt', 'r')
pcd_files = [x.strip() for x in f.readlines()]
pcd_files = [x for x in pcd_files if 'test61' in x or 'test50' in x]

save_dir = '/home/chrisxie/projects/ssc/example_images/'

i = 0
for pcd_filename in pcd_files:
    
    print(pcd_filename)
    
    ### Process .pcd file ###
    temp_idx = pcd_filename.split('/').index('OSD') # parse something like this: /data/OSD/OSD-0.2/pcd/learn44.pcd
    label_abs_path = '/'.join(pcd_filename.split('/')[temp_idx+1:])
    point_cloud = open3d.read_point_cloud(pcd_filename)
    
    # Fill in missing pixel values for RGB
    num_missing = 480*640 - np.asarray(point_cloud.colors).shape[0]
    filled_in_rgb_img = np.concatenate([np.asarray(point_cloud.colors), np.zeros((num_missing,3))])
    rgb_img = np.round(255 * filled_in_rgb_img.reshape(480,640,3)).astype(np.uint8)
        
    # Fill in missing xyz values
    num_missing = 480*640 - np.asarray(point_cloud.points).shape[0]
    filled_in_points = np.concatenate([np.asarray(point_cloud.points), np.zeros((num_missing,3))])
    xyz_img = np.asarray(filled_in_points).reshape(480,640,3)
    xyz_img[np.isnan(xyz_img)] = 0
    
    # Load label
    label_filename = pcd_filename.split('/')[-1].split('.pcd')[0] + '.png'
    label_filename = '/'.join(pcd_filename.split('/')[:-2]) + '-depth/annotation/' + label_filename
    label_img = cv2.imread(label_filename, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    label_img[label_img > 0] = label_img[label_img > 0] + 1 # so values are in [0, 2, 3, ...] (e.g. no table label)
    
    np.save(f'{save_dir}OSD_image_{i}.npy', {'rgb': rgb_img, 'xyz': xyz_img, 'label': label_img})
    i += 1




### OCID ###

pcd_files = ['/data/OCID-dataset/ARID20/table/top/seq07/pcd/result_2018-08-21-14-20-11.pcd',
             '/data/OCID-dataset/ARID10/table/top/mixed/seq14/pcd/result_2018-08-23-12-15-46.pcd',
            ]

save_dir = '/home/chrisxie/projects/ssc/example_images/'

i = 0
for pcd_filename in pcd_files:
    print(pcd_filename)

    ### Process .pcd file ###
    temp_idx = pcd_filename.split('/').index('OCID-dataset') # parse something like this: /data/OCID-dataset/YCB10/table/top/curved/seq36/pcd/result_2018-08-24-15-13-13.pcd
    label_abs_path = '/'.join(pcd_filename.split('/')[temp_idx+1:])
    point_cloud = open3d.read_point_cloud(pcd_filename)
    
    # Fill in missing pixel values for RGB
    num_missing = 480*640 - np.asarray(point_cloud.colors).shape[0]
    filled_in_rgb_img = np.concatenate([np.asarray(point_cloud.colors), np.zeros((num_missing,3))])
    rgb_img = np.round(255 * filled_in_rgb_img.reshape(480,640,3)).astype(np.uint8)
        
    # Fill in missing xyz values
    num_missing = 480*640 - np.asarray(point_cloud.points).shape[0]
    filled_in_points = np.concatenate([np.asarray(point_cloud.points), np.zeros((num_missing,3))])
    xyz_img = np.asarray(filled_in_points).reshape(480,640,3)
    xyz_img[np.isnan(xyz_img)] = 0
        
    # Load label
    label_filename = pcd_filename.split('/')[-1].split('.pcd')[0] + '.png'
    label_filename = '/'.join(pcd_filename.split('/')[:-2]) + '/label/' + label_filename
    label_img = cv2.imread(label_filename, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    label_img[label_img == 1] = 0
    label_img = (label_img.astype(int) - 1).clip(min=0).astype(np.uint8)
    
    np.save(f'{save_dir}OCID_image_{i}.npy', {'rgb': rgb_img, 'xyz': xyz_img, 'label': label_img})
    i += 1

"""