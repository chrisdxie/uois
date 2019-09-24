import math
from torch import nn
from torch.autograd import Function
import torch
import hough_voting_cuda


class HoughVotingFunction(Function):
    @staticmethod
    def forward(ctx, label, directions, skip_pixels, inlier_threshold, 
                angle_discretization, inlier_distance, percentage_threshold,
                object_center_kernel_radius):

        """
            label: [N x H x W] torch tensor of foreground mask. Values in {0,1}
            directions: [N x 2 x H x W] torch float tensor of center directions
            skip_pixels: (int) don't loop over all pixels, skip some for computational efficiency
            inlier_threshold: (float) minimum cosine similarity for inliers. value in [-1, 1]
            angle_discretization: (int) number of bins to discretize 360 degrees into
            inlier_distance: (int) maximum Euclidean distance of inliers to vote for centers
            percentage_threshold: (float) minimum percentage of voted angles to be considered an object center
            object_center_kernel_radius: (int) kernel size used for NMS for selecting local maximums
        """

        initial_masks, num_objects, object_center_indices = \
            hough_voting_cuda.hough_voting_forward(label, directions, skip_pixels, 
                                                   inlier_threshold, angle_discretization, inlier_distance, 
                                                   percentage_threshold, object_center_kernel_radius)

        # Process initial objects mask
        max_objects = initial_masks.shape[1]

        # Recall that object labels start at 2 (background=0, table=1)
        labels = torch.arange(2, max_objects+2, device=initial_masks.device, dtype=initial_masks.dtype)
        labels = labels.unsqueeze(-1).unsqueeze(-1) # Shape: [max_objects, 1, 1]
        initial_masks = torch.sum(labels * initial_masks, dim=1) # Shape: [N x H x W]

        # Compute object centers as y,x locations
        width = directions.shape[3]
        object_centers = torch.stack([object_center_indices / width,
                                      object_center_indices % width], dim=1) # Shape: [N x 2 x max_objects]. y,x location

        return initial_masks, num_objects, object_centers

    @staticmethod
    def backward(ctx, top_diff_box, top_diff_pose):
        return None, None, None, None, None, None, None, None, None, None


class HoughVoting(nn.Module):
    def __init__(self, skip_pixels=10, inlier_threshold=0.9, angle_discretization=30, 
                 inlier_distance=20, percentage_threshold=0.5, object_center_kernel_radius=10):
        super(HoughVoting, self).__init__()
        self.skip_pixels = skip_pixels
        self.inlier_threshold = inlier_threshold
        self.angle_discretization = angle_discretization
        self.inlier_distance = inlier_distance
        self.percentage_threshold = percentage_threshold
        self.object_center_kernel_radius = object_center_kernel_radius

    def forward(self, label, directions):
        return HoughVotingFunction.apply(label, directions, self.skip_pixels, 
                         self.inlier_threshold, self.angle_discretization, 
                         self.inlier_distance, self.percentage_threshold,
                         self.object_center_kernel_radius)

