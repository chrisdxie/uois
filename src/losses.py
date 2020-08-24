import torch
import torch.nn as nn

# My libraries
from . import cluster

BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2


class WeightedLoss(nn.Module):

    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.weighted = False

    def generate_weight_mask(self, mask, to_ignore=None):
        """ Generates a weight mask where pixel weights are inversely proportional to 
            how many pixels are in the class

            @param mask: a [N x ...] torch.FloatTensor with values in {0, 1, 2, ..., K+1}, where K is number of objects. {0,1} are background/table.
            @param to_ignore: a list of classes (integers) to ignore when creating mask

            @return: a torch.FloatTensor that is same shape as mask.
        """
        N = mask.shape[0]

        if self.weighted:

            # Compute pixel weights
            weight_mask = torch.zeros_like(mask).float() # Shape: [N x H x W]. weighted mean over pixels

            for i in range(N):

                unique_object_labels = torch.unique(mask[i])
                for obj in unique_object_labels: # e.g. [0, 1, 2, 5, 9, 10]. bg, table, 4 objects

                    if to_ignore is not None and obj in to_ignore:
                        continue

                    num_pixels = torch.sum(mask[i] == obj, dtype=torch.float)
                    weight_mask[i, mask[i] == obj] = 1 / num_pixels # inversely proportional to number of pixels

        else:
            weight_mask = torch.ones_like(mask) # mean over observed pixels
            if to_ignore is not None:
                for obj in to_ignore:
                    weight_mask[mask == obj] = 0

        return weight_mask


class CELossWeighted(WeightedLoss):
    """ Compute weighted CE loss with logits
    """

    def __init__(self, weighted=False):
        super(CELossWeighted, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target):
        """ Compute weighted cross entropy
            @param x: a [N x C x H x W] torch.FloatTensor of values
            @param target: a [N x H x W] torch.LongTensor of values
        """
        temp = self.CrossEntropyLoss(x, target) # Shape: [N x H x W]
        weight_mask = self.generate_weight_mask(target)
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask) 

        return loss


class CELossWeightedMasked(WeightedLoss):
    """ Compute weighted CE loss with logits
    """

    def __init__(self, weighted=False):
        super(CELossWeightedMasked, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target, fg_mask):
        """ Compute weighted cross entropy
            @param x: a [N x C x H x W] torch.FloatTensor of values
            @param target: a [N x H x W] torch.LongTensor of values
            @param fg_mask: a [N x H x W] torch.LongTensor of values in {0, 1, 2, ...}
        """
        temp = self.CrossEntropyLoss(x, target) # Shape: [N x H x W]
        weight_mask = self.generate_weight_mask(fg_mask, to_ignore=[0,1]) # ignore bg/table
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask) 

        return loss


def create_M_GT(foreground_labels):
    """ Create GT for Cross-Entropy Loss for M tensor.
        NOTE: no bg/table. obj index starts from 0.

        @param foreground_labels: a [H x W] torch.FloatTensor of values in {0, 1, 2, ..., K-1}
    """
    new_label = torch.zeros_like(foreground_labels)

    obj_index = 0
    for k in torch.unique(foreground_labels):

        if k in [BACKGROUND_LABEL, TABLE_LABEL]:
            continue

        # Objects get labels from 0, 1, 2, ..., K-1
        new_label[foreground_labels == k] = obj_index
        obj_index += 1

    return new_label.long()


class BCEWithLogitsLossWeighted(WeightedLoss):
    """ Compute weighted BCE loss with logits
    """
    def __init__(self, weighted=False):
        super(BCEWithLogitsLossWeighted, self).__init__()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target):
        """ Compute masked cosine similarity loss
            @param x: a [N x H x W] torch.FloatTensor of foreground logits
            @param target: a [N x H x W] torch.FloatTensor of values in [0, 1]
        """
        temp = self.BCEWithLogitsLoss(x, target) # Shape: [N x H x W]. values are in [0, 1]
        weight_mask = self.generate_weight_mask(target)
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask) 

        return loss

class SmoothL1LossWeighted(WeightedLoss):
    """ Compute weighted Smooth L1 loss
    """
    def __init__(self, weighted=False):
        super(SmoothL1LossWeighted, self).__init__()
        self.SmoothL1Loss = nn.SmoothL1Loss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target, mask=None):
        """ Compute masked cosine similarity loss
            @param x: a [N x C x H x W] torch.FloatTensor of values
            @param target: a [N x C x H x W] torch.FloatTensor of values
            @param mask: a [N x H x W] torch.FloatTensor with values in {0, 1, 2, ..., K+1}, where K is number of objects. {0,1} are background/table.
                                       Could also be None
        """
        temp = self.SmoothL1Loss(x, target).sum(dim=1) # Shape: [N x H x W]
        if mask is None:
            return torch.sum(temp) / temp.numel() # return mean

        weight_mask = self.generate_weight_mask(mask)
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask) 

        return loss

class ClusterLossWeighted(WeightedLoss):
    """ Compute Cluster loss
    """
    def __init__(self, delta, weighted=False):
        super(ClusterLossWeighted, self).__init__()
        self.weighted=weighted
        self.delta = delta

    def forward(self, x1, y1, x2, y2):
        """ Compute loss

            @param x1: a [M x D] torch.FloatTensor
            @param y1: a [M] torch.LongTensor
            @param x2: a [N x D] torch.FloatTensor
            @param y2: a [N] torch.LongTensor
            NOTE: N is NOT batch size
        """
        weight_vector_1 = self.generate_weight_mask(y1.unsqueeze(0))[0] # Shape: [M]
        weight_vector_2 = self.generate_weight_mask(y2.unsqueeze(0))[0] # Shape: [N]
        weight_matrix = torch.ger(weight_vector_1, weight_vector_2) # Shape: [M x N]
        indicator_matrix = (y1.unsqueeze(1) == y2.unsqueeze(0)).long() # Shape: [M x N]
        distance_matrix = cluster.euclidean_distances(x1,x2) # Shape: [M x N]

        positive_loss_matrix = indicator_matrix * distance_matrix**2

        negative_loss_matrix = (1 - indicator_matrix) * torch.clamp(self.delta - distance_matrix, min=0)**2

        return (weight_matrix * (positive_loss_matrix + negative_loss_matrix)).sum()




