import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import cv2
import json
import pybullet as p

from .util import utilities as util_
from . import data_augmentation

NUM_VIEWS_PER_SCENE = 7

BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2


###### Some utilities #####

def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


############# Synthetic Tabletop Object Dataset #############

class Tabletop_Object_Dataset(Dataset):
    """ Data loader for Tabletop Object Dataset
    """


    def __init__(self, base_dir, train_or_test, config):
        self.base_dir = base_dir
        self.config = config
        self.train_or_test = train_or_test

        # Get a list of all scenes
        self.scene_dirs = sorted(glob.glob(self.base_dir + '*/'))
        self.len = len(self.scene_dirs) * NUM_VIEWS_PER_SCENE

        self.name = 'TableTop'

        if 'v6' in self.base_dir:
            global OBJECTS_LABEL
            OBJECTS_LABEL = 4

    def __len__(self):
        return self.len

    def process_rgb(self, rgb_img):
        """ Process RGB image
                - random color warping
        """
        rgb_img = rgb_img.astype(np.float32)

        if self.config['use_data_augmentation']:
            # rgb_img = data_augmentation.random_color_warp(rgb_img)
            pass
        rgb_img = data_augmentation.standardize_image(rgb_img)

        return rgb_img

    def process_depth(self, depth_img):
        """ Process depth channel
                TODO: CHANGE THIS
                - change from millimeters to meters
                - cast to float32 data type
                - add random noise
                - compute xyz ordered point cloud
        """

        # millimeters -> meters
        depth_img = (depth_img / 1000.).astype(np.float32)

        # add random noise to depth
        if self.config['use_data_augmentation']:
            depth_img = data_augmentation.add_noise_to_depth(depth_img, self.config)
            # depth_img = data_augmentation.dropout_random_ellipses(depth_img, self.config)

        # Compute xyz ordered point cloud
        xyz_img = util_.compute_xyz(depth_img, self.config)
        if self.config['use_data_augmentation']:
            xyz_img = data_augmentation.add_noise_to_xyz(xyz_img, depth_img, self.config)

        return xyz_img

    def process_label_3D(self, foreground_labels, xyz_img, scene_description):
        """ Process foreground_labels

            @param foreground_labels: a [H x W] numpy array of labels
            @param xyz_img: a [H x W x 3] numpy array of xyz coordinates (in left-hand coordinate system)
            @param scene_description: a Python dictionary describing scene

            @return: foreground_labels
                     offsets: a [H x W x 2] numpy array of 2D directions. The i,j^th element has (y,x) direction to object center
        """

        # Any zero depth value will have foreground label set to background
        foreground_labels = foreground_labels.copy()
        foreground_labels[xyz_img[..., 2] == 0] = 0

        # Get inverse of camera extrinsics matrix. This is called "view_matrix" in OpenGL jargon
        view_num = scene_description['view_num']
        if view_num == 0: # bg-only image
            camera_dict = scene_description['views']['background']
        elif view_num == 1: # bg+table image
            key = 'background+tabletop' if 'v6' in self.base_dir else 'background+table'
            camera_dict = scene_description['views'][key] # table for TODv5, tabletop for TODv6
        else: # bg+table+objects image
            key = 'background+tabletop' if 'v6' in self.base_dir else 'background+table'
            camera_dict = scene_description['views'][key + '+objects'][view_num-2]
        view_matrix = p.computeViewMatrix(camera_dict['camera_pos'], 
                                          camera_dict['lookat_pos'], 
                                          camera_dict['camera_up_vector']
                                         )
        view_matrix = np.array(view_matrix).reshape(4,4, order='F')

        # Compute object centers and directions
        H, W = foreground_labels.shape
        offsets = np.zeros((H,W,3), dtype=np.float32)
        cf_3D_centers = np.zeros((100, 3), dtype=np.float32) # 100 max object centers
        for i, k in enumerate(np.unique(foreground_labels)):

            # Get mask
            mask = foreground_labels == k

            # For background/table, prediction direction should point towards origin
            if k in [BACKGROUND_LABEL, TABLE_LABEL]:
                offsets[mask, ...] = 0
                continue

            # Compute 3D object centers in camera frame
            ind = k - OBJECTS_LABEL
            wf_3D_center = np.array(scene_description['object_descriptions'][ind]['axis_aligned_bbox3D_center'])
            cf_3D_center = view_matrix.dot(np.append(wf_3D_center, 1.))[:3] # in OpenGL camera frame (right-hand system, z-axis pointing backward)
            cf_3D_center[2] = -1 * cf_3D_center[2] # negate z to get the left-hand system, z-axis pointing forward

            # If center isn't contained within the object, use point cloud average
            if cf_3D_center[0] < xyz_img[mask, 0].min() or \
               cf_3D_center[0] > xyz_img[mask, 0].max() or \
               cf_3D_center[1] < xyz_img[mask, 1].min() or \
               cf_3D_center[1] > xyz_img[mask, 1].max():
                cf_3D_center = xyz_img[mask, ...].mean(axis=0)

            # Get directions
            cf_3D_centers[i-2] = cf_3D_center
            object_center_offsets = (cf_3D_center - xyz_img).astype(np.float32) # Shape: [H x W x 3]

            # Add it to the labels
            offsets[mask, ...] = object_center_offsets[mask, ...]

        return offsets, cf_3D_centers

    def __getitem__(self, idx):

        cv2.setNumThreads(0) # some hack to make sure pyTorch doesn't deadlock. Found at https://github.com/pytorch/pytorch/issues/1355. Seems to work for me

        # Get scene directory
        scene_idx = idx // NUM_VIEWS_PER_SCENE
        scene_dir = self.scene_dirs[scene_idx]

        # Get view number
        view_num = idx % NUM_VIEWS_PER_SCENE

        # RGB image
        rgb_img_filename = scene_dir + f"rgb_{view_num:05d}.jpeg"
        rgb_img = cv2.cvtColor(cv2.imread(rgb_img_filename), cv2.COLOR_BGR2RGB)
        rgb_img = self.process_rgb(rgb_img)

        # Depth image
        depth_img_filename = scene_dir + f"depth_{view_num:05d}.png"
        depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH) # This reads a 16-bit single-channel image. Shape: [H x W]
        xyz_img = self.process_depth(depth_img)

        # Labels
        foreground_labels_filename = scene_dir + f"segmentation_{view_num:05d}.png"
        foreground_labels = util_.imread_indexed(foreground_labels_filename)
        scene_description_filename = scene_dir + "scene_description.txt"
        scene_description = json.load(open(scene_description_filename))
        scene_description['view_num'] = view_num

        center_offset_labels, object_centers = self.process_label_3D(foreground_labels, xyz_img, scene_description)

        label_abs_path = '/'.join(foreground_labels_filename.split('/')[-2:]) # Used for evaluation

        # Turn these all into torch tensors
        rgb_img = data_augmentation.array_to_tensor(rgb_img) # Shape: [3 x H x W]
        xyz_img = data_augmentation.array_to_tensor(xyz_img) # Shape: [3 x H x W]
        foreground_labels = data_augmentation.array_to_tensor(foreground_labels) # Shape: [H x W]
        center_offset_labels = data_augmentation.array_to_tensor(center_offset_labels) # Shape: [2 x H x W]
        object_centers = data_augmentation.array_to_tensor(object_centers) # Shape: [100 x 3]
        num_3D_centers = torch.tensor(np.count_nonzero(np.unique(foreground_labels) >= OBJECTS_LABEL))

        return {'rgb' : rgb_img,
                'xyz' : xyz_img,
                'foreground_labels' : foreground_labels,
                'center_offset_labels' : center_offset_labels,
                'object_centers' : object_centers, # This is gonna bug out because the dimensions will be different per frame
                'num_3D_centers' : num_3D_centers,
                'scene_dir' : scene_dir,
                'view_num' : view_num,
                'label_abs_path' : label_abs_path,
                }


def get_TOD_train_dataloader(base_dir, config, batch_size=8, num_workers=4, shuffle=True):

    config = config.copy()
    dataset = Tabletop_Object_Dataset(base_dir + 'training_set/', 'train', config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)

def get_TOD_test_dataloader(base_dir, config, batch_size=8, num_workers=4, shuffle=False):

    config = config.copy()
    dataset = Tabletop_Object_Dataset(base_dir + 'test_set/', 'test', config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)




############# RGB Images Dataset (Google Open Images) #############

class RGB_Objects_Dataset(Dataset):
    """ Data loader for Tabletop Object Dataset
    """


    def __init__(self, base_dir, start_list_file, train_or_test, config):
        self.base_dir = base_dir
        self.config = config
        self.train_or_test = train_or_test

        # Get a list of all instance labels
        f = open(base_dir + start_list_file)
        lines = [x.strip() for x in f.readlines()]
        self.starts = lines
        self.len = len(self.starts)

        self.name = 'RGB_Objects'

    def __len__(self):
        return self.len

    def pad_crop_resize(self, img, morphed_label, label):
        """ Crop the image around the label mask, then resize to 224x224
        """

        H, W, _ = img.shape

        # Get tight box around label/morphed label
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(label)
        _xmin, _ymin, _xmax, _ymax = util_.mask_to_tight_box(morphed_label)
        x_min = min(x_min, _xmin); y_min = min(y_min, _ymin); x_max = max(x_max, _xmax); y_max = max(y_max, _ymax)

        # Make bbox square
        x_delta = x_max - x_min
        y_delta = y_max - y_min
        if x_delta > y_delta:
            y_max = y_min + x_delta
        else:
            x_max = x_min + y_delta

        sidelength = x_max - x_min
        padding_percentage = np.random.beta(self.config['padding_alpha'], self.config['padding_beta'])
        padding_percentage = max(padding_percentage, self.config['min_padding_percentage'])
        padding = int(round(sidelength * padding_percentage))
        if padding == 0:
            print(f'Whoa, padding is 0... sidelength: {sidelength}, %: {padding_percentage}')
            padding = 25 # just make it 25 pixels

        # Pad and be careful of boundaries
        x_min = max(x_min - padding, 0)
        x_max = min(x_max + padding, W-1)
        y_min = max(y_min - padding, 0)
        y_max = min(y_max + padding, H-1)

        # Crop
        if (y_min == y_max) or (x_min == x_max):
            print('Fuck... something is wrong:', x_min, y_min, x_max, y_max)
            print(morphed_label)
            print(label)
        img_crop = img[y_min:y_max+1, x_min:x_max+1]
        morphed_label_crop = morphed_label[y_min:y_max+1, x_min:x_max+1]
        label_crop = label[y_min:y_max+1, x_min:x_max+1]

        # Resize
        img_crop = cv2.resize(img_crop, (224,224))
        morphed_label_crop = cv2.resize(morphed_label_crop, (224,224))
        label_crop = cv2.resize(label_crop, (224,224))

        return img_crop, morphed_label_crop, label_crop

    def transform(self, img, label):
        """ Process RGB image
                - standardize_image
                - random color warping
                - random horizontal flipping
        """

        img = img.astype(np.float32)

        # Data augmentation for mask
        morphed_label = label.copy()
        if self.config['use_data_augmentation']:
            if np.random.rand() < self.config['rate_of_morphological_transform']:
                morphed_label = data_augmentation.random_morphological_transform(morphed_label, self.config)
            if np.random.rand() < self.config['rate_of_translation']:
                morphed_label = data_augmentation.random_translation(morphed_label, self.config)
            if np.random.rand() < self.config['rate_of_rotation']:
                morphed_label = data_augmentation.random_rotation(morphed_label, self.config)

            sample = np.random.rand()
            if sample < self.config['rate_of_label_adding']:
                morphed_label = data_augmentation.random_add(morphed_label, self.config)
            elif sample < self.config['rate_of_label_adding'] + self.config['rate_of_label_cutting']:
                morphed_label = data_augmentation.random_cut(morphed_label, self.config)
                
            if np.random.rand() < self.config['rate_of_ellipses']:
                morphed_label = data_augmentation.random_ellipses(morphed_label, self.config)

        # Next, crop the mask with some padding, and resize to 224x224. Make sure to preserve the aspect ratio
        img_crop, morphed_label_crop, label_crop = self.pad_crop_resize(img, morphed_label, label)

        # Data augmentation for RGB
        # if self.config['use_data_augmentation']:
        #     img_crop = data_augmentation.random_color_warp(img_crop)
        img_crop = data_augmentation.standardize_image(img_crop)

        # Turn into torch tensors
        img_crop = data_augmentation.array_to_tensor(img_crop) # Shape: [3 x H x W]
        morphed_label_crop = data_augmentation.array_to_tensor(morphed_label_crop) # Shape: [H x W]
        label_crop = data_augmentation.array_to_tensor(label_crop) # Shape: [H x W]

        return img_crop, morphed_label_crop, label_crop

    def __getitem__(self, idx):

        cv2.setNumThreads(0) # some hack to make sure pyTorch doesn't deadlock. Found at https://github.com/pytorch/pytorch/issues/1355. Seems to work for me

        # Get label filename
        label_filename = self.starts[idx]

        label = cv2.imread(str(os.path.join(self.base_dir, 'Labels', label_filename))) # Shape: [H x W x 3]
        label = label[..., 0] == 255 # Turn it into a {0,1} binary mask with shape: [H x W]
        label = label.astype(np.uint8)

        # find corresponding image file
        img_file = label_filename.split('_')[0] + '.jpg'
        img = cv2.imread(str(os.path.join(self.base_dir, 'Images', img_file)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # These might not be the same size. resize them to the smaller one
        if label.shape[0] < img.shape[0]:
            new_size = label.shape[::-1] # (W, H)
        else:
            new_size = img.shape[:2][::-1]
        label = cv2.resize(label, new_size)
        img = cv2.resize(img, new_size)

        img_crop, morphed_label_crop, label_crop = self.transform(img, label)

        return {
            'rgb' : img_crop,
            'initial_masks' : morphed_label_crop,
            'labels' : label_crop
        }

def get_RGBO_train_dataloader(base_dir, config, batch_size=8, num_workers=4, shuffle=True):

    dataset = RGB_Objects_Dataset(base_dir, config['starts_file'], 'train', config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)


# Synthetic RGB dataset for training RGB Refinement Network
class Synthetic_RGB_Objects_Dataset(RGB_Objects_Dataset):
    """ Data loader for Tabletop Object Dataset
    """

    def __init__(self, base_dir, train_or_test, config):
        self.base_dir = base_dir
        self.config = config
        self.train_or_test = train_or_test

        # Get a list of all scenes
        self.scene_dirs = sorted(glob.glob(self.base_dir + '*/'))
        self.len = len(self.scene_dirs) * 5 # only 5 images with objects in them

        self.name = 'Synth_RGB_Objects'

    def __getitem__(self, idx):

        cv2.setNumThreads(0) # some hack to make sure pyTorch doesn't deadlock. Found at https://github.com/pytorch/pytorch/issues/1355. Seems to work for me

        # Get scene directory
        scene_idx = idx // 5
        scene_dir = self.scene_dirs[scene_idx]

        # Get view number
        view_num = idx % 5 + 2 # objects start at rgb_00002.jpg

        # Label
        foreground_labels_filename = scene_dir + f"segmentation_{view_num:05d}.png"
        label_abs_path = '/'.join(foreground_labels_filename.split('/')[-2:]) # Used for evaluation
        foreground_labels = util_.imread_indexed(foreground_labels_filename)

        # Grab a random object and use that mask
        obj_ids = np.unique(foreground_labels)
        if obj_ids[0] == 0:
            obj_ids = obj_ids[1:] # get rid of background
        if obj_ids[0] == 1:
            obj_ids = obj_ids[1:] # get rid of table

        num_pixels = 1; num_pixel_tries = 0
        while num_pixels < 2:

            if num_pixel_tries > 100:
                print("ERROR. Pixels too small. Choosing a new image.")
                print(scene_dir, view_num, num_pixels, obj_ids, np.unique(foreground_labels))

                # Choose a new image to use instead
                new_idx = np.random.randint(0, self.len)
                return self.__getitem__(new_idx)

            obj_id = np.random.choice(obj_ids)
            label = (foreground_labels == obj_id).astype(np.uint8)
            num_pixels = np.count_nonzero(label)

            num_pixel_tries += 1

        # RGB image
        img_filename = scene_dir + f"rgb_{view_num:05d}.jpeg"
        img = cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB)

        # Processing
        img_crop, morphed_label_crop, label_crop = self.transform(img, label)

        return {
            'rgb' : img_crop,
            'initial_masks' : morphed_label_crop,
            'labels' : label_crop,
            'label_abs_path' : label_abs_path,
        }

def get_Synth_RGBO_train_dataloader(base_dir, config, batch_size=8, num_workers=4, shuffle=True):

    dataset = Synthetic_RGB_Objects_Dataset(base_dir + 'training_set/','train', config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)

