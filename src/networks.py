import torch
import torch.nn as nn

# My Libraries
from .util import utilities as util_


def maxpool2x2(input, ksize=2, stride=2):
    """2x2 max pooling"""
    return nn.MaxPool2d(ksize, stride=stride)(input)

class Conv2d_GN_ReLU(nn.Module):
    """ Implements a module that performs 
            conv2d + groupnorm + ReLU + 

        Assumes kernel size is odd
    """

    def __init__(self, in_channels, out_channels, num_groups, ksize=3, stride=1):
        super(Conv2d_GN_ReLU, self).__init__()
        padding = 0 if ksize < 2 else ksize//2
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=ksize, stride=stride, 
                               padding=padding, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)

        return out

class Conv2d_GN_ReLUx2(nn.Module):
    """ Implements a module that performs 
            conv2d + groupnorm + ReLU + 
            conv2d + groupnorm + ReLU
            (and a possible downsampling operation)

        Assumes kernel size is odd
    """

    def __init__(self, in_channels, out_channels, num_groups, ksize=3, stride=1):
        super(Conv2d_GN_ReLUx2, self).__init__()
        self.layer1 = Conv2d_GN_ReLU(in_channels, out_channels, 
                                     num_groups, ksize=ksize, stride=stride)
        self.layer2 = Conv2d_GN_ReLU(out_channels, out_channels, 
                                     num_groups, ksize=ksize, stride=stride)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        return out

# Code adapted from: https://github.com/sacmehta/ESPNet/blob/master/train/Model.py
class ESPModule(nn.Module):
    """ This class is the ESP module from ESP-Netv2.

        Changes:
            - first convlution is a normal 3x3 conv
            - We use GroupNorm instead of BatchNorm, and ReLU instead of PReLU
    """
    def __init__(self, in_channels, out_channels, num_groups, ksize=1, add=True):
        """
            @param in_channels: number of input channels
            @param out_channels: number of output channels
            @param num_groups: number of groups for GroupNorm
            @param ksize: kernel size for original conv
            @param add: if true, add a residual connection through identity operation
        """
        super().__init__()
        n = int(out_channels / 5)
        n1 = out_channels - 4 * n

        c1_padding = 0 if ksize < 2 else ksize//2
        self.conv1 = nn.Conv2d(in_channels, n, 
                               kernel_size=ksize, stride=1, 
                               padding=c1_padding, bias=False)

        self.dilated1 = nn.Conv2d(n, n1, 
                                  kernel_size=3, stride=1, 
                                  padding=1, bias=False, dilation=1)
        self.dilated2 = nn.Conv2d(n, n, 
                                  kernel_size=3, stride=1, 
                                  padding=2, bias=False, dilation=2)
        self.dilated4 = nn.Conv2d(n, n, 
                                  kernel_size=3, stride=1, 
                                  padding=4, bias=False, dilation=4)
        self.dilated8 = nn.Conv2d(n, n, 
                                  kernel_size=3, stride=1, 
                                  padding=8, bias=False, dilation=8)
        self.dilated16 = nn.Conv2d(n, n, 
                                   kernel_size=3, stride=1, 
                                   padding=16, bias=False, dilation=16)

        self.gn = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.add = add

    def forward(self, input):

        # Reduce
        output1 = self.conv1(input)

        # Split and Transform
        d1 = self.dilated1(output1)
        d2 = self.dilated2(output1)
        d4 = self.dilated4(output1)
        d8 = self.dilated8(output1)
        d16 = self.dilated16(output1)

        # Heirarchical Feature Fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # Merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        if self.add:
            combine = input + combine
        output = self.gn(combine)
        output = self.relu(output)

        return output

class Upsample_Concat_Conv2d_GN_ReLU(nn.Module):
    """ Implements a module that performs
            Upsample (reduction: conv2d + groupnorm + ReLU + bilinear_sampling) +
            concat + conv2d + groupnorm + ReLU 

        The Upsample operation consists of a Conv2d_GN_ReLU that reduces the channels by 2,
            followed by bilinear sampling

        Note: in_channels is number of channels of ONE of the inputs to the concatenation

    """
    def __init__(self, in_channels, out_channels, num_groups, ksize=3, stride=1):
        super(Upsample_Concat_Conv2d_GN_ReLU, self).__init__()
        self.channel_reduction_layer = Conv2d_GN_ReLU(in_channels, in_channels//2, num_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_gn_relu = Conv2d_GN_ReLU(in_channels, out_channels, num_groups)

    def forward(self, x1, x2):
        x1 = self.channel_reduction_layer(x1)
        x1 = self.upsample(x1)
        out = torch.cat([x1, x2], dim=1) # Concat on channels dimension
        out = self.conv_gn_relu(out)

        return out

class Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(nn.Module):
    """ Implements a module that performs
            Upsample (reduction: conv2d + groupnorm + ReLU + bilinear_sampling) +
            concat + conv2d + groupnorm + ReLU 
        for the U-Net decoding architecture with an arbitrary number of encoders

        The Upsample operation consists of a Conv2d_GN_ReLU that reduces the channels by 2,
            followed by bilinear sampling

        Note: in_channels is number of channels of ONE of the inputs to the concatenation

    """
    def __init__(self, in_channels, out_channels, num_groups, num_encoders, ksize=3, stride=1):
        super(Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch, self).__init__()
        self.channel_reduction_layer = Conv2d_GN_ReLU(in_channels, in_channels//2, num_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_gn_relu = Conv2d_GN_ReLU(int(in_channels//2 * (num_encoders+1)), out_channels, num_groups)

    def forward(self, x, skips):
        """ Forward module

            @param skips: a list of intermediate skip-layer torch tensors from each encoder
        """
        x = self.channel_reduction_layer(x)
        x = self.upsample(x)
        out = torch.cat([x] + skips, dim=1) # Concat on channels dimension
        out = self.conv_gn_relu(out)

        return out




################## Network Definitions ##################

class UNet_Encoder(nn.Module):
    
    def __init__(self, input_channels, feature_dim):
        super(UNet_Encoder, self).__init__()
        self.ic = input_channels
        self.fd = feature_dim
        self.build_network()
        
    def build_network(self):
        """ Build encoder network
            Uses a U-Net-like architecture
        """

        ### Encoder ###
        self.layer1 = Conv2d_GN_ReLUx2(self.ic, self.fd, self.fd)
        self.layer2 = Conv2d_GN_ReLUx2(self.fd, self.fd*2, self.fd)
        self.layer3 = Conv2d_GN_ReLUx2(self.fd*2, self.fd*4, self.fd)
        self.layer4 = Conv2d_GN_ReLUx2(self.fd*4, self.fd*8, self.fd)
        self.last_layer = Conv2d_GN_ReLU(self.fd*8, self.fd*16, self.fd)


    def forward(self, images):

        x1 = self.layer1(images)
        mp_x1 = maxpool2x2(x1)
        x2 = self.layer2(mp_x1)
        mp_x2 = maxpool2x2(x2)
        x3 = self.layer3(mp_x2)
        mp_x3 = maxpool2x2(x3)
        x4 = self.layer4(mp_x3)
        mp_x4 = maxpool2x2(x4)
        x5 = self.last_layer(mp_x4)

        return x5, [x1, x2, x3, x4]


class UNet_Decoder(nn.Module):

    def __init__(self, num_encoders, feature_dim):
        super(UNet_Decoder, self).__init__()
        self.ne = num_encoders
        self.fd = feature_dim
        self.build_network()

    def build_network(self):
        """ Build a decoder network
            Uses a U-Net-like architecture
        """

        # Fusion layer
        self.fuse_layer = Conv2d_GN_ReLU(self.fd*16 * self.ne, self.fd*16, self.fd, ksize=1)

        # Decoding
        self.layer1 = Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.fd*16, self.fd*8, self.fd, self.ne)
        self.layer2 = Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.fd*8, self.fd*4, self.fd, self.ne)
        self.layer3 = Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.fd*4, self.fd*2, self.fd, self.ne)
        self.layer4 = Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.fd*2, self.fd, self.fd, self.ne)

        # Final layer
        self.layer5 = Conv2d_GN_ReLU(self.fd, self.fd, self.fd)

        # This puts features everywhere, not just nonnegative orthant
        self.last_conv = nn.Conv2d(self.fd, self.fd, kernel_size=3,
                                   stride=1, padding=1, bias=True)

    def forward(self, encoder_list):
        """ Forward module

            @param encoder_list: a list of tuples
                                 each tuple includes 2 elements:
                                    - output of encoder: an [N x C x H x W] torch tensor
                                    - list of intermediate outputs: a list of 4 torch tensors

        """

        # Apply fusion layer to the concatenation of encoder outputs
        out = torch.cat([x[0] for x in encoder_list], dim=1) # Concatenate on channels dimension
        out = self.fuse_layer(out)

        out = self.layer1(out, [x[1][3] for x in encoder_list])
        out = self.layer2(out, [x[1][2] for x in encoder_list])
        out = self.layer3(out, [x[1][1] for x in encoder_list])
        out = self.layer4(out, [x[1][0] for x in encoder_list])
        out = self.layer5(out)

        out = self.last_conv(out)

        return out


class UNetESP_Encoder(nn.Module):
    
    def __init__(self, input_channels, feature_dim):
        super(UNetESP_Encoder, self).__init__()
        self.ic = input_channels
        self.fd = feature_dim
        self.build_network()
        
    def build_network(self):
        """ Build encoder network
            Uses a U-Net-like architecture
        """

        ### Encoder ###
        self.layer1 = Conv2d_GN_ReLUx2(self.ic, self.fd, self.fd)
        self.layer2 = Conv2d_GN_ReLUx2(self.fd, self.fd*2, self.fd)
        self.layer3a = Conv2d_GN_ReLU(self.fd*2, self.fd*4, self.fd)
        self.layer3b = ESPModule(self.fd*4, self.fd*4, self.fd, ksize=3)
        self.layer4a = Conv2d_GN_ReLU(self.fd*4, self.fd*8, self.fd)
        self.layer4b = ESPModule(self.fd*8, self.fd*8, self.fd, ksize=3)
        self.last_layer = Conv2d_GN_ReLU(self.fd*8, self.fd*16, self.fd)


    def forward(self, images):

        x1 = self.layer1(images)
        mp_x1 = maxpool2x2(x1)
        x2 = self.layer2(mp_x1)
        mp_x2 = maxpool2x2(x2)
        x3 = self.layer3a(mp_x2)
        x3 = self.layer3b(x3)
        mp_x3 = maxpool2x2(x3)
        x4 = self.layer4a(mp_x3)
        x4 = self.layer4b(x4)
        mp_x4 = maxpool2x2(x4)
        x5 = self.last_layer(mp_x4)

        return x5, [x1, x2, x3, x4]


class UNetESP_Decoder(nn.Module):
    """ Like UNet decoder above, but assumes 1 encoder. Also uses ESP module
    """
    def __init__(self, feature_dim):
        super(UNetESP_Decoder, self).__init__()
        self.fd = feature_dim
        self.build_network()

    def build_network(self):
        """ Build a decoder network
            Uses a U-Net-like architecture
        """

        # Fusion layer
        self.fuse_layer = ESPModule(self.fd*16, self.fd*16, self.fd, ksize=1)

        # Decoding
        self.layer1 = Upsample_Concat_Conv2d_GN_ReLU(self.fd*16, self.fd*8, self.fd)
        self.layer2 = Upsample_Concat_Conv2d_GN_ReLU(self.fd*8, self.fd*4, self.fd)
        self.layer3 = Upsample_Concat_Conv2d_GN_ReLU(self.fd*4, self.fd*2, self.fd)
        self.layer4 = Upsample_Concat_Conv2d_GN_ReLU(self.fd*2, self.fd, self.fd)

        # Final layer
        self.layer5 = Conv2d_GN_ReLU(self.fd, self.fd, self.fd)

        # This puts features everywhere, not just nonnegative orthant
        self.last_conv = nn.Conv2d(self.fd, self.fd, kernel_size=3,
                                   stride=1, padding=1, bias=True)

    def forward(self, encoder_outputs):
        """ Forward module

            @param encoder_outputs: a tuple of 2 elements:
                                    - output of encoder: an [N x C x H x W] torch tensor
                                    - list of intermediate outputs: a list of 4 torch tensors

        """

        # Apply fusion layer to the concatenation of encoder outputs
        out = self.fuse_layer(encoder_outputs[0])

        out = self.layer1(out, encoder_outputs[1][3])
        out = self.layer2(out, encoder_outputs[1][2])
        out = self.layer3(out, encoder_outputs[1][1])
        out = self.layer4(out, encoder_outputs[1][0])

        out = self.layer5(out)
        out = self.last_conv(out)

        return out