import torch
import torch.nn as nn

BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2

class CELossWeighted(nn.Module):
    """ Compute weighted CE loss with logits
    """

    def __init__(self):
        super(CELossWeighted, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, target):
        """ Compute weighted cross entropy

            @param x: a [N x C x H x W] torch.FloatTensor of values
            @param target: a [N x H x W] torch.LongTensor of values
        """
        temp = self.CrossEntropyLoss(x, target) # Shape: [N x H x W]

        # Compute pixel weights
        weight_mask = torch.zeros_like(target).float() # Shape: [N x H x W]. weighted mean over pixels
        unique_object_labels = torch.unique(target)
        for obj in unique_object_labels:
            num_pixels = torch.sum(target == obj, dtype=torch.float)
            weight_mask[target == obj] = 1 / num_pixels # inversely proportional to number of pixels

        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask) 
        return loss

class CosineSimilarityLossWithMask(nn.Module):
    """ Compute Cosine Similarity loss
    """
    def __init__(self, weighted=False):
        super(CosineSimilarityLossWithMask, self).__init__()
        self.CosineSimilarity = nn.CosineSimilarity(dim=1)
        self.weighted = weighted

    def forward(self, x, target, mask=None):
        """ Compute masked cosine similarity loss

            @param x: a [N x C x H x W] torch.FloatTensor of values
            @param target: a [N x C x H x W] torch.FloatTensor of values
            @param mask: a [N x H x W] torch.FloatTensor with values in {0, 1, 2, ..., K+1}, where K is number of objects. {0,1} are background/table.
                                       Could also be None
        """
        temp = .5 * (1 - self.CosineSimilarity(x, target)) # Shape: [N x H x W]. values are in [0, 1]
        if mask is None:
            return torch.sum(temp) / target.numel() # return mean

        # Compute tabletop objects mask
        binary_object_mask = (mask.clamp(0,2).long() == OBJECTS_LABEL) # Shape: [N x H x W]

        if torch.sum(binary_object_mask) > 0:
            if self.weighted:
                # Compute pixel weights
                weight_mask = torch.zeros_like(mask) # Shape: [N x H x W]. weighted mean over pixels
                unique_object_labels = torch.unique(mask)
                unique_object_labels = unique_object_labels[unique_object_labels >= 2]
                for obj in unique_object_labels:
                    num_pixels = torch.sum(mask == obj, dtype=torch.float)
                    weight_mask[mask == obj] = 1 / num_pixels # inversely proportional to number of pixels
            else:
                weight_mask = binary_object_mask.float() # mean over observed pixels
            loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask) 
        else:
            print("all gradients are 0...")
            loss = torch.tensor(0., dtype=torch.float, device=x.device) # just 0. all gradients will be 0

        bg_mask = ~binary_object_mask
        if torch.sum(bg_mask) > 0:
            bg_loss = 0.1 * torch.sum(temp * bg_mask.float()) / torch.sum(bg_mask.float())
        else:
            bg_loss = torch.tensor(0., dtype=torch.float, device=x.device) # just 0

        return loss + bg_loss

class BCEWithLogitsLossWeighted(nn.Module):
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

        if self.weighted:
            # Compute pixel weights
            weight_mask = torch.zeros_like(target) # Shape: [N x H x W]. weighted mean over pixels
            unique_object_labels = torch.unique(target) # Should be {0, 1}
            for obj in unique_object_labels:
                num_pixels = torch.sum(target == obj, dtype=torch.float)
                weight_mask[target == obj] = 1 / num_pixels # inversely proportional to number of pixels
        else:
            weight_mask = torch.ones_like(target) # mean over observed pixels
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask) 

        return loss