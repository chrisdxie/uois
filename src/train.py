from time import time
import torch
import numpy as np

from abc import ABC, abstractmethod

# tensorboard stuff
from torch.utils.tensorboard import SummaryWriter

# my libraries
from .util import utilities as util_
from . import losses as ls
from . import cluster

BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2

NUM_GPUS = torch.cuda.device_count()



def smart_random_sample_indices(X, Y, num_seeds):
    """ Helper function to sample seeds for mean shift training

        @param predicted_centers: a [N x 3] torch.FloatTensor
        @param Y: a [N] torch.LongTensor with values in {2, ... K+1}
        @param num_seeds: int
    """

    unique_obj_labels = torch.unique(Y)
    num_objects = unique_obj_labels.shape[0]

    indices = torch.zeros(0, dtype=torch.long, device=X.device)

    num_seeds_per_obj = int(np.ceil(num_seeds / num_objects))
    for k in unique_obj_labels:
        label_indices = torch.where(Y == k)[0]
        randperm = torch.randperm(label_indices.shape[0])
        inds = label_indices[randperm[:num_seeds_per_obj]]
        indices = torch.cat([indices, inds], dim=0)

    X_I = X[indices, :] # Shape: [num_seeds x 3]
    Y_I = Y[indices] # Shape: [num_seeds]

    return X_I, Y_I


def hill_climb_one_iter(Z, X, sigmas):
    """ Runs one iteration of GBMS hill climbing algorithm
        The seeds climb the distribution given by the KDE of X
        Note: X is not edited by this method
    
        @param Z: a [m x d] torch.FloatTensor of seeds
        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param sigmas: a [1 x n] torch.FloatTensor of sigmas OR a Python float
    """

    W = cluster.gaussian_kernel(Z, X, sigmas) # Shape: [m x n]
    Q = W / W.sum(dim=1, keepdim=True) # Shape: [m x n]
    Z = torch.mm(Q, X)

    return Z

class Trainer(ABC):

    def __init__(self, model_wrapper, config):
        self.model_wrapper = model_wrapper
        self.device = self.model_wrapper.device
        self.config = config
        self.setup()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def save(self, name=None, save_dir=None):
        """ Save optimizer state, epoch/iter nums, loss information

            Also save model state
        """

        # Save optimizer stuff
        checkpoint = {
            'iter_num' : self.iter_num,
            'epoch_num' : self.epoch_num,
            'infos' : self.infos,
        }

        checkpoint['optimizer'] = self.optimizer.state_dict()

        if save_dir is None:
            save_dir = self.config['tb_directory']
        if name is None:
            filename = save_dir + self.__class__.__name__ + '_' \
                                + self.model_wrapper.__class__.__name__ \
                                + '_iter' + str(self.iter_num) \
                                + '_checkpoint.pth'
        else:
            filename = save_dir + name + '_checkpoint.pth'
        torch.save(checkpoint, filename)


        # Save model stuff
        filename = save_dir + self.model_wrapper.__class__.__name__ \
                            + '_iter' + str(self.iter_num) \
                            + '_' + str(self.model_wrapper.config['feature_dim']) + 'c' \
                            + '_checkpoint.pth'
        self.model_wrapper.save(filename)

    def load(self, opt_filename, model_filename):
        """ Load optimizer state, epoch/iter nums, loss information

            Also load model state
        """

        # Load optimizer stuff
        checkpoint = torch.load(opt_filename)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded optimizer")

        self.iter_num = checkpoint['iter_num']
        self.epoch_num = checkpoint['epoch_num']
        self.infos = checkpoint['infos']


        # Load model stuff
        self.model_wrapper.load(model_filename)

class DSNTrainer(Trainer):

    def setup(self):

        # Initialize stuff
        self.epoch_num = 1
        self.iter_num = 1
        self.infos = dict()

        # Initialize optimizer
        model_config = self.model_wrapper.model.parameters()
        self.optimizer = torch.optim.Adam(model_config, lr=self.config['lr'])

        if self.config['load']:
            self.load(self.config['opt_filename'], self.config['model_filename'])

        # Losses
        foreground_loss = ls.CELossWeighted(weighted=True)
        center_offset_loss = ls.SmoothL1LossWeighted(weighted=True)
        separation_loss = ls.CELossWeightedMasked(weighted=True)
        cluster_loss = ls.ClusterLossWeighted(self.config['delta'], weighted=True)
        self.losses = {
            'fg_loss' : foreground_loss,
            'co_loss' : center_offset_loss,
            'sep_loss'  : separation_loss,
            'cl_loss' : cluster_loss,
        }

        # Tensorboard stuff
        self.tb_writer = SummaryWriter(self.config['tb_directory'],
                                       flush_secs=self.config['flush_secs'])   
                                       
    def train(self, num_epochs, data_loader):

        # Some stuff to keep track of
        batch_time = util_.AverageMeter()
        data_time = util_.AverageMeter()
        total_losses = util_.AverageMeter()
        fg_losses = util_.AverageMeter()
        center_offset_losses = util_.AverageMeter()
        cluster_losses = util_.AverageMeter()
        separation_losses = util_.AverageMeter()
        end = time()

        # Training mode
        self.model_wrapper.train_mode()

        for epoch_iter in range(num_epochs):
            for i, batch in enumerate(data_loader):

                if self.iter_num >= self.config['max_iters']:
                    print("Reached maximum number of iterations...")
                    break

                # Send everything to GPU
                self.model_wrapper.send_batch_to_device(batch)

                # Get labels
                foreground_labels = batch['foreground_labels'] # Shape: [N x H x W]
                center_offset_labels = batch['center_offset_labels'] # Shape: [N x 2 x H x W]
                object_centers = batch['object_centers'] # Shape: [N x 100 x 3]

                # measure data loading time
                data_time.update(time() - end)
                N, H, W = foreground_labels.shape

                # This is (potentially) in parallel
                fg_logits, center_offsets = self.model_wrapper.model(batch['xyz'])

                ### Foreground Loss ###
                fg_masks = foreground_labels.clamp(0,2).long()
                fg_loss = self.losses['fg_loss'](fg_logits, fg_masks)

                ### Center Prediction Loss ###
                center_offset_loss = self.losses['co_loss'](center_offsets, center_offset_labels, foreground_labels)

                separation_loss = torch.tensor(0.).to(self.device)
                cluster_loss = torch.tensor(0.).to(self.device)
                L = self.config['max_GMS_iters']
                for j in range(N):

                    fg_mask_j = fg_masks[j] == OBJECTS_LABEL
                    if torch.sum(fg_mask_j) == 0:
                        continue

                    ### Separation Loss ###
                    predicted_centers = batch['xyz'][j] + center_offsets[j]

                    # Cross-Entropy Loss for M. Only run on FG pixels
                    gt_centers = object_centers[j, :batch['num_3D_centers'][j]]
                    M_logits = self.model_wrapper.construct_M_logits(predicted_centers, gt_centers)
                    M_gt = ls.create_M_GT(foreground_labels[j])
                    separation_loss = separation_loss + \
                                      self.losses['sep_loss'](M_logits.unsqueeze(0),
                                                              M_gt.unsqueeze(0), 
                                                              foreground_labels[j].unsqueeze(0))

                    ### Cluster loss ###
                    # Note: the meanshift is spread across GPUs to spread memory across
                    X = predicted_centers.permute(1,2,0)
                    X_fg = X[fg_mask_j][::self.model_wrapper.config['subsample_factor'], ...] # Shape: [num_fg_pixels//subsample_factor x 3]
                    Y_fg = foreground_labels[j, fg_mask_j][::self.model_wrapper.config['subsample_factor']] # Shape: [num_fg_pixels//subsample_factor]
                    X_fg = X_fg.to(f'cuda:{j % NUM_GPUS}'); Y_fg = Y_fg.to(f'cuda:{j % NUM_GPUS}')
                    X_I, Y_I = smart_random_sample_indices(X_fg, Y_fg, self.config['num_seeds_training'])
                    for l in range(L):
                        X_I = hill_climb_one_iter(X_I, X_fg, self.model_wrapper.config['sigma'])
                        cluster_loss = cluster_loss + self.losses['cl_loss'](X_I, Y_I, X_fg, Y_fg).to(self.device)

                # Weight the frame-wise losses by batch size
                separation_loss = separation_loss / N
                cluster_loss = cluster_loss / (N * L)

                # Total loss. Note: foreground loss is always computed/backpropagated
                loss = self.config['lambda_fg'] * fg_loss + \
                       self.config['lambda_co'] * center_offset_loss + \
                       self.config['lambda_sep'] * separation_loss + \
                       self.config['lambda_cl'] * cluster_loss


                ### Gradient descent ###
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                total_losses.update(loss.item(), N)
                fg_losses.update(fg_loss.item(), N)
                center_offset_losses.update(center_offset_loss.item(), N)
                cluster_losses.update(cluster_loss.item(), N)
                separation_losses.update(separation_loss.item(), N)

                # Record some information about this iteration
                batch_time.update(time() - end)
                end = time()

                # Record information every x iterations
                if self.iter_num % self.config['iter_collect'] == 0:
                    info = {'iter_num': self.iter_num,
                            'Batch Time': round(batch_time.avg, 3),
                            'Data Time': round(data_time.avg, 3),
                            'loss': round(total_losses.avg, 7),
                            'FG loss': round(fg_losses.avg, 7),
                            'Center Offset loss': round(center_offset_losses.avg, 7),
                            'Cluster loss': round(cluster_losses.avg, 7),
                            'Separation loss' : round(separation_losses.avg, 7),
                            }
                    self.infos[self.iter_num] = info

                    # Tensorboard stuff
                    self.tb_writer.add_scalar('Total Loss', info['loss'], self.iter_num)
                    self.tb_writer.add_scalar('Loss/Foreground', info['FG loss'], self.iter_num)
                    self.tb_writer.add_scalar('Loss/Center Offset', info['Center Offset loss'], self.iter_num)
                    self.tb_writer.add_scalar('Loss/Cluster', info['Cluster loss'], self.iter_num)
                    self.tb_writer.add_scalar('Loss/Separation', info['Separation loss'], self.iter_num)
                    self.tb_writer.add_scalar('Time/per iter', info['Batch Time'], self.iter_num)
                    self.tb_writer.add_scalar('Time/data fetch', info['Data Time'], self.iter_num)

                    # Reset meters
                    batch_time = util_.AverageMeter()
                    data_time = util_.AverageMeter()
                    total_losses = util_.AverageMeter()
                    fg_losses = util_.AverageMeter()
                    center_offset_losses = util_.AverageMeter()
                    cluster_losses = util_.AverageMeter()
                    separation_losses = util_.AverageMeter()
                    end = time()

                self.iter_num += 1

            self.epoch_num += 1

class RRNTrainer(Trainer):

    def setup(self):

        # Initialize stuff
        self.epoch_num = 1
        self.iter_num = 1
        self.infos = dict()

        # Initialize optimizer
        model_config = self.model_wrapper.model.parameters()
        self.optimizer = torch.optim.Adam(model_config, lr=self.config['lr'])

        if self.config['load']:
            self.load(self.config['opt_filename'], self.config['model_filename'])

        # Losses
        foreground_loss = ls.BCEWithLogitsLossWeighted(weighted=True)
        self.losses = {
            'fg_loss' : foreground_loss,
        }

        # Tensorboard stuff
        self.tb_writer = SummaryWriter(self.config['tb_directory'],
                                       flush_secs=self.config['flush_secs'])   

    def train(self, num_epochs, data_loader):

        # Some stuff to keep track of
        batch_time = util_.AverageMeter()
        data_time = util_.AverageMeter()
        total_losses = util_.AverageMeter()
        end = time()

        # Training mode
        self.model_wrapper.train_mode()

        for epoch_iter in range(num_epochs):
            for i, batch in enumerate(data_loader):

                if self.iter_num >= self.config['max_iters']:
                    print("Reached maximum number of iterations...")
                    break

                # Send everything to GPU
                self.model_wrapper.send_batch_to_device(batch)

                # Get labels
                labels = batch['labels'].float() # Shape: [N x H x W]

                # measure data loading time
                data_time.update(time() - end)
                N, H, W = labels.shape

                # Apply the model
                logits = self.model_wrapper.model(batch) # Shape: [N x 3 x H x W]

                # Apply the loss
                loss = self.losses['fg_loss'](logits, labels)


                ### Gradient descent ###
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                total_losses.update(loss.item(), N)

                # Record some information about this iteration
                batch_time.update(time() - end)
                end = time()

                # Record information every x iterations
                if self.iter_num % self.config['iter_collect'] == 0:
                    info = {'iter_num': self.iter_num,
                            'Batch Time': round(batch_time.avg, 3),
                            'Data Time': round(data_time.avg, 3),
                            'loss': round(total_losses.avg, 7),
                            }
                    self.infos[self.iter_num] = info

                    # Tensorboard stuff
                    self.tb_writer.add_scalar('Total Loss', info['loss'], self.iter_num)
                    self.tb_writer.add_scalar('Time/per iter', info['Batch Time'], self.iter_num)
                    self.tb_writer.add_scalar('Time/data fetch', info['Data Time'], self.iter_num)

                    # Reset meters
                    batch_time = util_.AverageMeter()
                    data_time = util_.AverageMeter()
                    total_losses = util_.AverageMeter()
                    end = time()

                self.iter_num += 1

            self.epoch_num += 1




