import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from abc import ABC, abstractmethod

# my libraries
from .util import utilities as util_
from . import networks
from . import losses
from . import cluster

BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2

# dictionary keys returned by dataloaders that you don't want to send to GPU
dont_send_to_device = ['scene_dir', 
                       'view_num', 
                       'subset', 
                       'supporting_plane', 
                       'label_abs_path',
                      ]

class NetworkWrapper(ABC):

    def __init__(self, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config.copy()

        # Build network and losses
        self.setup()

    @abstractmethod
    def setup(self):
        pass

    def train_mode(self):
        """ Put all modules into train mode
        """
        self.model.train()

    def eval_mode(self):
        """ Put all modules into eval mode
        """
        self.model.eval()

    def send_batch_to_device(self, batch):
        for key in batch.keys():
            if key in dont_send_to_device:
                continue
            if len(batch[key]) == 0: # can happen if a modality (e.g. RGB) is not loaded
                continue
            batch[key] = batch[key].to(self.device)

    def save(self, filename):
        """ Save the model as a checkpoint
        """
        checkpoint = {'model' : self.model.state_dict()}
        torch.save(checkpoint, filename)

    def load(self, filename):
        """ Load the model checkpoint
        """
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model'])
        print(f"Loaded {self.__class__.__name__} model")


class DepthSeedingNetwork(nn.Module):

    def __init__(self, encoder, decoder, fg_module, cd_module):
        super(DepthSeedingNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fg_module = fg_module
        self.cd_module = cd_module

    def forward(self, xyz_img):
        """ Forward pass using entire DSN

            @param xyz_img: a [N x 3 x H x W] torch.FloatTensor of xyz depth images

            @return: fg_logits: a [N x 3 x H x W] torch.FloatTensor of background/table/object logits
                     center_offsets: a [N x 2 x H x W] torch.FloatTensor of center direction predictions
        """
        features = self.decoder(self.encoder(xyz_img))
        fg_logits = self.fg_module(features)
        center_offsets = self.cd_module(features)

        return fg_logits, center_offsets


class DSNWrapper(NetworkWrapper):

    def setup(self):
        """ Setup model, losses, optimizers, misc
        """

        # Encoder
        self.encoder = networks.UNetESP_Encoder(input_channels=3,
                                                feature_dim=self.config['feature_dim'])

        # Decoder
        self.decoder = networks.UNetESP_Decoder(feature_dim=self.config['feature_dim'])

        # A 1x1 conv layer that goes from embedded features to logits for 3 classes: background (0), table (1), objects (2)
        self.foreground_module = nn.Conv2d(self.config['feature_dim'], 3, 
                                           kernel_size=1, stride=1, 
                                           padding=0, bias=False)

        # A 1x1 conv layer that goes from embedded features to 3d center offsets
        self.center_direction_module = nn.Conv2d(self.config['feature_dim'], 3,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=False)

        # Whole model, for nn.DataParallel
        self.model = DepthSeedingNetwork(self.encoder, self.decoder, 
                                         self.foreground_module, 
                                         self.center_direction_module,
                                        )

        print("Let's use", torch.cuda.device_count(), "GPUs for DSN!")
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        
    def cluster(self, xyz_img, offsets, fg_mask):
        """ Run mean shift clustering algorithm on predicted 3D centers

            @param xyz_img: a [3 x H x W] torch.FloatTensor
            @param offsets: a [3 x H x W] torch.FloatTensor of center offsets
            @param fg_mask: a [H x W] torch.IntTensor of foreground. 1 for foreground object, 0 for background/table

            @return: a [H x W] torch.LongTensor
        """
        clustered_img = torch.zeros_like(fg_mask, dtype=torch.long)
        if torch.sum(fg_mask) == 0: # No foreground pixels to cluster
            return clustered_img, torch.zeros((0,3), device=self.device)

        predicted_centers = xyz_img + offsets
        predicted_centers = predicted_centers.permute(1,2,0) # Shape: [H x W x 3]

        # Cluster the predicted centers (ONLY the predictions of foreground pixels)
        ms = cluster.GaussianMeanShift(
                max_iters=self.config['max_GMS_iters'],
                epsilon=self.config['epsilon'], 
                sigma=self.config['sigma'], 
                num_seeds=self.config['num_seeds'],
                subsample_factor=self.config['subsample_factor']
             )
        cluster_labels = ms.mean_shift_smart_init(predicted_centers[fg_mask])

        # Reshape clustered labels back to [H x W]
        clustered_img[fg_mask] = cluster_labels + OBJECTS_LABEL

        # Get cluster centers
        uniq_cluster_centers = ms.uniq_cluster_centers
        uniq_labels = ms.uniq_labels + OBJECTS_LABEL

        # Get rid of small clusters
        uniq_counts = torch.zeros_like(uniq_labels)
        for j, label in enumerate(uniq_labels):
            uniq_counts[j] = torch.sum(clustered_img == label)
        valid_indices = []
        for j, label in enumerate(uniq_labels):
            if uniq_counts[j] < self.config['min_pixels_thresh']:
                continue
            valid_indices.append(j)
        valid_indices = np.array(valid_indices)

        new_cl_img = torch.zeros_like(clustered_img)
        if valid_indices.shape[0] > 0:
            uniq_cluster_centers = uniq_cluster_centers[valid_indices, :]

            # relabel clustered_img to match uniq_cluster_centers
            new_label = OBJECTS_LABEL
            for j in valid_indices:
                new_cl_img[clustered_img == uniq_labels[j]] = new_label
                new_label += 1

        else: # no valid indices, all clusters were erased
            uniq_cluster_centers = torch.zeros((0,3), dtype=torch.float, device=self.device)
        clustered_img = new_cl_img

        return clustered_img, uniq_cluster_centers

    def construct_M_logits(self, predicted_centers, object_centers):
        """ Construct the logits of tensor M given the object centers, and the predicted centers
            for a SINGLE frame.

            Note: We do NOT multiply by foreground here. This is mostly used for training

            @param predicted_centers: a [3 x H x W] torch.FloatTensor of center predictions
            @param object_centers: a [num_objects x 3] torch.FloatTensor. 
            @param fg_logits: a [3 x H x W] torch.FloatTensor of foreground logits

            @return: a [num_objects x H x W] torch.FloatTensor
        """

        # Compute expontiated distances, then apply LogSoftmax
        distances = torch.norm(predicted_centers.unsqueeze(0) - object_centers.unsqueeze(-1).unsqueeze(-1), dim=1) # Shape: [num_obj x H x W]

        return -self.config['tau'] * distances

    def run_on_batch(self, batch):
        """ Run algorithm with 3D voting on batch of images in eval mode

            @param batch: a dictionary with the following keys:
                            - xyz: a [N x 3 x H x W] torch.FloatTensor

            @return fg_mask: a [N x H x W] torch.LongTensor with values in {0, 1, 2}
                    center_offsets: a [N x 3 x H x W] torch.FloatTensor
                    object_centers: a list of [num_objects x 3] torch.IntTensor. This list has length N
                    initial_masks: a [N x H x W] torch.IntTensor. Note: Initial masks has values in {0, 2, 3, ...}. No table
        """

        self.eval_mode()
        self.send_batch_to_device(batch)

        N, _, H, W = batch['xyz'].shape
        initial_masks = torch.empty((N, H, W), dtype=torch.long, device=self.device)
        object_centers = []

        with torch.no_grad():

            # Apply model
            fg_logits, center_offsets = self.model(batch['xyz'])

            # Foreground
            fg_probs = F.softmax(fg_logits, dim=1) # Shape: [N x 3 x H x W]
            fg_mask = torch.argmax(fg_probs, dim=1) # Shape: [N x H x W]

            # Clustering, construct M
            for i in range(N):
                clustered_img, cluster_centers = self.cluster(batch['xyz'][i], 
                                                              center_offsets[i], 
                                                              fg_mask[i] == OBJECTS_LABEL)
                initial_masks[i] = clustered_img
                object_centers.append(cluster_centers)

        return fg_mask, center_offsets, object_centers, initial_masks




class RegionRefinementNetwork(nn.Module):

    def __init__(self, encoder, decoder, fg_module):
        super(RegionRefinementNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fg_module = fg_module

    def forward(self, input_dict):
        """ Forward pass using entire RRN
            @param input_dict: A dictionary of torch tensors of different modalities.
                               e.g. keys could include: rgb, xyz
            @return: a [N x H x W] torch tensor of foreground logits
        """
        rgb = input_dict['rgb'] # Shape: [N x 3 x H x W], where H = W = 224
        initial_masks = input_dict['initial_masks'].unsqueeze(1) # Shape: [N x 1 x H x W]
        network_input = torch.cat([rgb, initial_masks], dim=1) # Shape: [N x 4 x H x W]
        features = self.decoder([self.encoder(network_input)])
        return self.fg_module(features)[:,0,:,:]

class RRNWrapper(NetworkWrapper):

    def setup(self):
        """ Setup model, losses, optimizers, misc
        """

        # Encoder
        self.encoder = networks.UNet_Encoder(input_channels=4,
                                             feature_dim=self.config['feature_dim'])

        # Decoder
        self.decoder = networks.UNet_Decoder(num_encoders=1, 
                                             feature_dim=self.config['feature_dim'])

        # A 1x1 conv layer that goes from embedded features to logits for foreground
        self.foreground_module = nn.Conv2d(self.config['feature_dim'], 1, 
                                           kernel_size=1, stride=1, 
                                           padding=0, bias=False)

        # Whole model, for nn.DataParallel
        self.model = RegionRefinementNetwork(self.encoder, self.decoder, self.foreground_module)
        
        print("Let's use", torch.cuda.device_count(), "GPUs for RRN!")
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def run_on_batch(self, batch, threshold=0.5):
        """ Run algorithm on batch of images in eval mode
            @param batch: a dictionary with the following keys:
                            - rgb: a [N x 3 x H x W] torch.FloatTensor
                            - initial_masks: a [N x H x W] torch.FloatTensor
        """

        self.eval_mode()
        self.send_batch_to_device(batch)

        with torch.no_grad():

            logits = self.model(batch) # Shape: [N x H x W]
            probs = torch.sigmoid(logits) # Shape: [N x H x W]
            masks = probs > threshold 

        return masks




class UOISNet3D(object):
    """ Class to encapsulate both Depth Seeding Network and RGB Refinement Network. """

    def __init__(self, config, dsn_filename, dsn_config, rrn_filename, rrn_config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.dsn = DSNWrapper(dsn_config)
        self.dsn.load(dsn_filename)

        self.rrn = RRNWrapper(rrn_config)
        self.rrn.load(rrn_filename)

        self.config = config

    def rrn_preprocess(self, rgb_img, initial_masks):
        """ Pad, Crop, and Resize to prepare input to RRN.

            @param rgb_img: a [3 x H x W] torch.FloatTensor
            @param initial_masks: a [H x W] torch tensor

            @return: a dictionary: {'rgb' : rgb_crops, 'initial_masks' : mask_crops}
                     a dictionary: {str(mask_id) : [x_min, y_min, x_max, y_max] for each mask_id}
        """
        _, H, W = rgb_img.shape

        # Dictionary to save crop indices
        crop_indices = {}

        mask_ids = torch.unique(initial_masks)
        mask_ids = mask_ids[mask_ids >= OBJECTS_LABEL]

        rgb_crops = torch.zeros((mask_ids.shape[0], 3, 224, 224), device=self.device)
        mask_crops = torch.zeros((mask_ids.shape[0], 224, 224), device=self.device)

        for index, mask_id in enumerate(mask_ids):
            mask = (initial_masks == mask_id).float() # Shape: [H x W]

            # crop the masks/rgb to 224x224 with some padding, save it as "initial_masks"
            x_min, y_min, x_max, y_max = util_.mask_to_tight_box(mask)
            x_padding = int(torch.round((x_max - x_min).float() * self.config['padding_percentage']).item())
            y_padding = int(torch.round((y_max - y_min).float() * self.config['padding_percentage']).item())

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

        batch = {'rgb' : rgb_crops, 'initial_masks' : mask_crops}
        return batch, crop_indices


    def rrn_postprocess(self, refined_crops, crop_indices, xyz_img):
        """ Resize the masks to original size.
            Paint the masks in ordered by distance from camera.

            @param refined_crops: a [num_masks x resized_H x resized_W] torch.FloatTensor
            @param crop_indices: a dictionary {str(mask_id) : [x_min, y_min, x_max, y_max] for each mask_id}
            @param xyz_img: a [3 x H x W] torch.FloatTensor

            @return: a [H x W] torch tensor of resized masks
        """
        refined_masks = torch.zeros_like(xyz_img[0,...]) # Shape: [H x W]
        mask_ids = crop_indices.keys()

        sorted_mask_ids = []
        for index, mask_id in enumerate(mask_ids):

            # Resize back to original size
            x_min, y_min, x_max, y_max = crop_indices[mask_id]
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
            x_min, y_min, x_max, y_max = crop_indices[mask_id]
            orig_H = y_max - y_min + 1
            orig_W = x_max - x_min + 1
            mask = refined_crops[index].unsqueeze(0).unsqueeze(0).float()
            resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0,0]

            # Set refined mask
            h_idx, w_idx = torch.nonzero(resized_mask).t()
            refined_masks[y_min:y_max+1, x_min:x_max+1][h_idx, w_idx] = mask_id

        return refined_masks

    def process_initial_masks(self, batch, initial_masks, object_centers):
        """ Process the initial masks:
                - open/close morphological transform
                - closest connected component to object center
            @param batch: a dictionary with the following keys:
                            - rgb: a [N x 3 x H x W] torch.FloatTensor
                            - xyz: a [N x 3 x H x W] torch.FloatTensor
            @param initial_masks: a [N x H x W] torch.IntTensor. Note: Initial masks has values in [0, 2, 3, ...]. No table
            @param object_centers: a list of [2 x num_objects] torch.IntTensor. This list has length N

            @return: the processed initial masks [N x H x W]
        """
        N, H, W = initial_masks.shape

        # Bring some tensors to numpy for processing
        initial_masks = initial_masks.cpu().numpy()
        for i in range(N):
            object_centers[i] = object_centers[i].cpu().numpy()
        xyz_imgs = batch['xyz'].cpu().numpy().transpose(0,2,3,1) # Shape: [N x H x W x 3]

        # Open/close morphology stuff
        if self.config['use_open_close_morphology']:

            for i in range(N):

                # Get object ids. Remove background (0)
                obj_ids = np.unique(initial_masks[i])
                if obj_ids[0] == 0:
                    obj_ids = obj_ids[1:]

                # For each object id, open/close the masks
                for obj_id in obj_ids:
                    mask = (initial_masks[i] == obj_id) # Shape: [H x W]

                    ksize = self.config['open_close_morphology_ksize'] # 9
                    opened_mask = cv2.morphologyEx(mask.astype(np.uint8), 
                                                   cv2.MORPH_OPEN, 
                                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))
                    opened_closed_mask = cv2.morphologyEx(opened_mask,
                                                          cv2.MORPH_CLOSE,
                                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))

                    h_idx, w_idx = np.nonzero(mask)
                    initial_masks[i, h_idx, w_idx] = 0
                    h_idx, w_idx = np.nonzero(opened_closed_mask)
                    initial_masks[i, h_idx, w_idx] = obj_id

        # Largest Connected Component
        if self.config['use_largest_connected_component']:

            pixel_indices = util_.build_matrix_of_indices(H, W)
            for i in range(N):
                
                # Get object ids. Remove background (0)
                obj_ids = np.unique(initial_masks[i])
                obj_ids = obj_ids[obj_ids >= OBJECTS_LABEL]

                # Loop over each object and run largest connected component
                for obj_index, obj_id in enumerate(obj_ids):
                    orig_mask = initial_masks[i] == obj_id
                    largest_component_mask = util_.largest_connected_component(orig_mask)
                    initial_masks[i][orig_mask] = 0
                    initial_masks[i][largest_component_mask] = obj_id

        initial_masks = torch.from_numpy(initial_masks).to(self.device)

        return initial_masks


    def run_on_batch(self, batch):
        """ Run algorithm on batch of images in eval mode
            @param batch: a dictionary with the following keys:
                            - rgb: a [N x 3 x H x W] torch.FloatTensor
                            - xyz: a [N x 3 x H x W] torch.FloatTensor
            @param final_close_morphology: If True, then run open/close morphology after refining mask.
                                           This typically helps a synthetically-trained RRN
        """
        N, _, H, W = batch['rgb'].shape

        # DSN. Note: this will send "batch" to device (e.g. GPU)
        fg_masks, center_offsets, object_centers, initial_masks = self.dsn.run_on_batch(batch)

        # IMP
        initial_masks = self.process_initial_masks(batch, initial_masks, object_centers)

        # RRN
        refined_masks = self.refine_with_RRN(batch, initial_masks)

        return fg_masks, center_offsets, initial_masks, refined_masks


    def refine_with_RRN(self, batch, initial_masks):
        """ Run refinement with RRN

            @param batch: a dictionary with the following keys:
                            - rgb: a [N x 3 x H x W] torch.FloatTensor
                            - xyz: a [N x 3 x H x W] torch.FloatTensor
            @param initial_masks: a [N x H x W] torch.IntTensor. Note: Initial masks has values in {0, 2, 3, ...}

            @return: a [N x H x W] torch.IntTensor. Note: Initial masks has values in {0, 2, 3, ...}
        """
        N, H, W = initial_masks.shape

        # Data structure to hold everything at end
        refined_masks = torch.zeros_like(initial_masks)
        for i in range(N):

            im_batch, crop_indices = self.rrn_preprocess(batch['rgb'][i], initial_masks[i])

            # Run the RGB Refinement Network
            N_masks = len(crop_indices)
            refined_crops = torch.zeros_like(im_batch['initial_masks'])
            step_size = 20
            for b in range(0, N_masks, step_size):
                refined_crops[b:b+step_size] = self.rrn.run_on_batch({
                                                    'rgb' : im_batch['rgb'][b:b+step_size],
                                                    'initial_masks' : im_batch['initial_masks'][b:b+step_size],
                                               })
            refined_masks[i,...] = self.rrn_postprocess(refined_crops, crop_indices, batch['xyz'][i])

        # Open/close morphology stuff, for synthetically-trained RRN
        if self.config['final_close_morphology']:
            refined_masks = refined_masks.cpu().numpy() # to CPU

            for i in range(N):

                # Get object ids. Remove background (0)
                obj_ids = np.unique(refined_masks[i])
                if obj_ids[0] == 0:
                    obj_ids = obj_ids[1:]

                # For each object id, open/close the masks
                for obj_id in obj_ids:
                    mask = (refined_masks[i] == obj_id) # Shape: [H x W]

                    ksize = self.config['open_close_morphology_ksize'] # 9
                    opened_mask = cv2.morphologyEx(mask.astype(np.uint8), 
                                                   cv2.MORPH_OPEN, 
                                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))
                    opened_closed_mask = cv2.morphologyEx(opened_mask,
                                                          cv2.MORPH_CLOSE,
                                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))

                    h_idx, w_idx = np.nonzero(mask)
                    refined_masks[i, h_idx, w_idx] = 0
                    h_idx, w_idx = np.nonzero(opened_closed_mask)
                    refined_masks[i, h_idx, w_idx] = obj_id

            refined_masks = torch.from_numpy(refined_masks).to(self.device) # back to GPU

        # Get rid of small clusters. Set them to background. Rest of objects in {2, 3, ..., K+1}
        for i in range(N):

            mapping = {}
            curr_obj_id = OBJECTS_LABEL
            uniq_labels = torch.unique(refined_masks[i])
            uniq_labels = uniq_labels[uniq_labels >= OBJECTS_LABEL]

            for label in uniq_labels:
                mask = refined_masks[i] == label
                if torch.sum(mask) < self.dsn.config['min_pixels_thresh']:
                    refined_masks[i,mask] = 0. # Set to background
                else:
                    mapping[label] = curr_obj_id
                    curr_obj_id += 1

            new_mask = torch.zeros_like(refined_masks[i])
            for label in mapping:
                new_mask[refined_masks[i] == label] = mapping[label]
            refined_masks[i] = new_mask

        return refined_masks
