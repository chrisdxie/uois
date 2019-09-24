import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import cv2

# My libraries
from .util import utilities as util_
from . import data_augmentation

NUM_VIEWS_PER_SCENE = 7

###### Some utilities #####

def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def compute_xyz(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters.

        If focal lengths fx,fy are stored in the camera_params dictionary, use that.
        Else, assume camera_params contains parameters used to generate synthetic data (e.g. fov, near, far, etc)

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used 
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

    indices = util_.build_matrix_of_indices(camera_params['img_height'], camera_params['img_width'])
    indices[..., 0] = np.flipud(indices[..., 0]) # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    
    return xyz_img




############# Synthetic Tabletop Object Dataset #############

class Tabletop_Object_Dataset(Dataset):
    """ Data loader for Tabletop Object Dataset
    """


    def __init__(self, base_dir, train_or_test, params):
        self.base_dir = base_dir
        self.params = params
        self.train_or_test = train_or_test

        # Get a list of all scenes
        self.scene_dirs = sorted(glob.glob(self.base_dir + '*/'))
        self.len = len(self.scene_dirs) * NUM_VIEWS_PER_SCENE

        self.name = 'TableTop'

    def __len__(self):
        return self.len

    def process_rgb(self, rgb_img):
        """ Process RGB image
                - random color warping
        """
        rgb_img = rgb_img.astype(np.float32)
        rgb_img = data_augmentation.standardize_image(rgb_img)

        return rgb_img

    def process_depth(self, depth_img, seg_img):
        """ Process depth channel
                - change from millimeters to meters
                - cast to float32 data type
                - add random noise
                - compute xyz ordered point cloud
        """

        # millimeters -> meters
        depth_img = (depth_img / 1000.).astype(np.float32)

        # add random noise to depth
        if self.params['use_data_augmentation']:
            depth_img = data_augmentation.add_noise_to_depth(depth_img, self.params)
            depth_img = data_augmentation.dropout_random_ellipses(depth_img, self.params)

        # Compute xyz ordered point cloud and add noise
        xyz_img = compute_xyz(depth_img, self.params)
        if self.params['use_data_augmentation']:
            xyz_img = data_augmentation.add_noise_to_xyz(xyz_img, depth_img, self.params)

        return xyz_img

    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
                     direction_labels: a [H x W x 2] numpy array of 2D directions. The i,j^th element has (y,x) direction to object center
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels

        # Compute object centers and directions
        H, W = foreground_labels.shape
        direction_labels = np.stack([np.ones((H,W)), np.zeros((H, W))], axis=-1).astype(np.float32) # Shape: [H x W x 2]
        pixel_indices = util_.build_matrix_of_indices(H, W)
        for k in np.unique(foreground_labels):

            if k in [0, 1]: # background, table
                continue

            # Get object mask
            object_mask = foreground_labels == k

            # Get average of all pixel indices in mask
            center = np.mean(pixel_indices[object_mask, :], axis=0) # Shape: [2]. y_center, x_center

            # Get directions
            object_center_directions = (center - pixel_indices).astype(np.float32) # Shape: [H x W x 2]
            object_center_directions = object_center_directions / np.maximum(np.linalg.norm(object_center_directions, axis=2, keepdims=True), 1e-10)

            # Add it to the labels
            direction_labels[object_mask] = object_center_directions[object_mask]

        return foreground_labels, direction_labels

    def __getitem__(self, idx):

        cv2.setNumThreads(0) # some hack to make sure pyTorch doesn't deadlock. Found at https://github.com/pytorch/pytorch/issues/1355

        # Get scene directory
        scene_idx = idx // NUM_VIEWS_PER_SCENE
        scene_dir = self.scene_dirs[scene_idx]

        # Get view number
        view_num = idx % NUM_VIEWS_PER_SCENE

        # Label
        foreground_labels_filename = scene_dir + f"segmentation_{view_num:05d}.png"
        label_abs_path = '/'.join(foreground_labels_filename.split('/')[-2:]) # Used for evaluation
        foreground_labels = util_.imread_indexed(foreground_labels_filename)
        foreground_labels, direction_labels = self.process_label(foreground_labels)

        # RGB image
        rgb_img_filename = scene_dir + f"rgb_{view_num:05d}.jpeg"
        rgb_img = cv2.cvtColor(cv2.imread(rgb_img_filename), cv2.COLOR_BGR2RGB)
        rgb_img = self.process_rgb(rgb_img)

        # Depth image
        if self.train_or_test == 'train':
            depth_img_filename = scene_dir + f"depth_{view_num:05d}.png"
        elif self.train_or_test == 'test':
            depth_img_filename = scene_dir + f"depth_noisy_{view_num:05d}.png"
        depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH) # This reads a 16-bit single-channel image. Shape: [H x W]
        xyz_img = self.process_depth(depth_img, foreground_labels)

        # Turn these all into torch tensors
        rgb_img = data_augmentation.array_to_tensor(rgb_img) # Shape: [3 x H x W]
        xyz_img = data_augmentation.array_to_tensor(xyz_img) # Shape: [3 x H x W]
        foreground_labels = data_augmentation.array_to_tensor(foreground_labels) # Shape: [H x W]
        direction_labels = data_augmentation.array_to_tensor(direction_labels) # Shape: [2 x H x W]

        return {'rgb' : rgb_img,
                'xyz' : xyz_img,
                'foreground_labels' : foreground_labels,
                'direction_labels' : direction_labels,
                'scene_dir' : scene_dir,
                'view_num' : view_num,
                'label_abs_path' : label_abs_path,
                }


def get_TOD_train_dataloader(base_dir, params, batch_size=8, num_workers=4, shuffle=True):

    dataset = Tabletop_Object_Dataset(base_dir + 'training_set/', 'train', params)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)

def get_TOD_test_dataloader(base_dir, params, batch_size=8, num_workers=4, shuffle=False):

    params = params.copy()
    params['use_data_augmentation'] = False
    dataset = Tabletop_Object_Dataset(base_dir + 'test_set/', 'test', params)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)




############# RGB Images Dataset (Google Open Images) #############

class RGB_Objects_Dataset(Dataset):
    """ Data loader for RGB Objects Dataset
    """

    def __init__(self, base_dir, start_list_file, train_or_test, params):
        self.base_dir = base_dir
        self.params = params
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
        padding_percentage = np.random.beta(self.params['padding_alpha'], self.params['padding_beta'])
        padding_percentage = max(padding_percentage, self.params['min_padding_percentage'])
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
            print('Whoa... something is wrong:', x_min, y_min, x_max, y_max)
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
        """ Data augmentation for RGB image and label
                - RGB
                    - Image standardization
                - Label
                    - Morphological transformation
                    - rotation/translation
                    - adding/cutting
                    - random ellipses
        """

        img = img.astype(np.float32)

        # Data augmentation for mask
        morphed_label = label.copy()
        if np.random.rand() < self.params['rate_of_morphological_transform']:
            morphed_label = data_augmentation.random_morphological_transform(morphed_label, self.params)
        if np.random.rand() < self.params['rate_of_translation']:
            morphed_label = data_augmentation.random_translation(morphed_label, self.params)
        if np.random.rand() < self.params['rate_of_rotation']:
            morphed_label = data_augmentation.random_rotation(morphed_label, self.params)

        sample = np.random.rand()
        if sample < self.params['rate_of_label_adding']:
            morphed_label = data_augmentation.random_add(morphed_label, self.params)
        elif sample < self.params['rate_of_label_adding'] + self.params['rate_of_label_cutting']:
            morphed_label = data_augmentation.random_cut(morphed_label, self.params)
            
        if np.random.rand() < self.params['rate_of_ellipses']:
            morphed_label = data_augmentation.random_ellipses(morphed_label, self.params)

        # Next, crop the mask with some padding, and resize to 224x224. Make sure to preserve the aspect ratio
        img_crop, morphed_label_crop, label_crop = self.pad_crop_resize(img, morphed_label, label)

        # Data processing for RGB
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

def get_RGBO_train_dataloader(base_dir, params, batch_size=8, num_workers=4, shuffle=True):

    dataset = RGB_Objects_Dataset(base_dir, params['starts_file'], 'train', params)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)




############# Synthetic RGB Objects Dataset (Tabletop Objects Dataset) #############
class Synthetic_RGB_Objects_Dataset(RGB_Objects_Dataset):
    """ Data loader for Synthetic RGB Objects Dataset
    """

    def __init__(self, base_dir, train_or_test, params):
        self.base_dir = base_dir
        self.params = params
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

def get_Synth_RGBO_train_dataloader(base_dir, params, batch_size=8, num_workers=4, shuffle=True):

    dataset = Synthetic_RGB_Objects_Dataset(base_dir + 'training_set/','train', params)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)


