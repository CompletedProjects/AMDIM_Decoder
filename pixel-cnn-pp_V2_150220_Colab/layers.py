from utils import *
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn
import numpy as np

class nin(nn.Module): # Nawid- Network in network - additional gated ResNet blocks with 1x1 convolution between regular convolution blocks that grow receptive field
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))# Nawid -Weight normalisation of a linear unit and the performs the linear transformation where the output dimension is dim_out
        self.dim_out = dim_out # Nawid - Output dimension

    def forward(self, x): # Nawid - Overall this is used to change the number of chanels with a linear transformation I believe
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """
        # TODO : try with original ordering
        x = x.permute(0, 2, 3, 1) # Nawid - Change the ordering of layers
        shp = [int(y) for y in x.size()] # Nawid - Obtains the shape
        out = self.lin_a(x.contiguous().view(shp[0]*shp[1]*shp[2], shp[3])) # Nawid - Performs a linear layer where shp[0]*shp[1]* shp[2] is the number of examples and shp[3] is the dimension which is being transformed
        shp[-1] = self.dim_out # Nawid - Changes last dimension to the output dimension
        out = out.view(shp) # Nawid - Change the shape of the output
        return out.permute(0, 3, 1, 2) # Nawid -Change the ordering of layers - II believe it changes it to batch size, channels


class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1),
                    shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__() # Nawid - Padding seems to agree with TF version of official implementation

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride) # Nawid - Convolution object
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad  = nn.ZeroPad2d((int((filter_size[1] - 1) / 2), # pad left
                                  int((filter_size[1] - 1) / 2), # pad right
                                  filter_size[0] - 1,            # pad top
                                  0) )                           # pad down

        if norm == 'weight_norm':
            self.conv == wn(self.conv) # Nawid-  Normalise the convolution object
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down : # Nawid - If shift output down, then the x is shifted downwards which is performed by down_shift method in the utils and x is given certain padding
            self.down_shift = lambda x : down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        x = self.pad(x) # Nawid-  Pads the image
        x = self.conv(x) # Nawid- Performs convolution
        x = self.bn(x) if self.norm == 'batch_norm' else x # Nawid - Performs batch normalisation
        return self.down_shift(x) if self.shift_output_down else x # Nawid - Down shifts x


class down_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1)):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride,
                                            output_padding=1)) # Nawid - Deconvolution object which is weight normalised
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()] # Nawid - Obtains all the dimensions from the deconvolution
        return x[:, :, :(xs[2] - self.filter_size[0] + 1),
                 int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))] # Nawid-


class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                    shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0)) # Nawid -Adds Padding to the left and the top ( no padding to the bottom)
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride) # Nawid - Convolution object
        self.shift_output_right = shift_output_right # Nawid - Whether the output should be shifted
        self.norm = norm

        if norm == 'weight_norm':
            self.conv == wn(self.conv) #Nawid - Weighnormalisation on the convolution
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right :
            self.right_shift = lambda x : right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0))) # Nawid - Performs right shift and padding

    def forward(self, x):
        x = self.pad(x) # Nawid - Padding
        x = self.conv(x) # Nawid - Convolution
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x # Nawid - Output which is potentially right shifted


class down_right_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                    shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size,
                                                stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x) # Nawid- Performs the deconvolution
        xs = [int(y) for y in x.size()] # Nawid -
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):, :(xs[3] - self.filter_size[1] + 1)]
        return x

'''
skip connection parameter : 0 = no skip connection
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size # Nawid - This is due to concat relu I believe
'''
class gated_resnet(nn.Module):# Nawid - I think this represents both of the gated resnets. If there is no auxillary variable present, then it represents the vertical gated resnet. If there is an auxillary variable present, then it can represent the horizontal layer
    def __init__(self, num_filters, conv_op, latent_dim, nonlinearity=concat_elu, skip_connection=0): # Nawid - Added a term for the latent dimensionality of the latent vector
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection # Nawid - Decides how to set the skip connection
        self.nonlinearity = nonlinearity # Nawid - Choose the non-linearity which is concat elu
        self.conv_input = conv_op(2 * num_filters, num_filters) # cuz of concat elu
        # Nawid - The number of inputs is 2* hidden feature maps and the outputs is equal to the num_filters
        if skip_connection != 0 :
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters) #Nawid - The first parameter is equal to the input dimensions and the next parameter is equal to the output dimensions. nin ( network in network) is used to ensure that the dimensions of the auxillary variable and the other dimension are the same


        self.hw = nin(latent_dim,2*num_filters) # Nawid - Initialises the network in a network 1 x1 convolution

            #self.hw = nn.Linear(h.size()[-1], 2*num_filters,bias=False)


        self.dropout = nn.Dropout2d(0.5) # Nawid- Performs dropout
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters) # Nawid - Convolution where the shapes are the same.

    def forward(self, og_x, a=None, h = None):
        x = self.conv_input(self.nonlinearity(og_x)) # Nawid - Performs the concat_elu which doubles the size and conv_input then backs it back to the original size - I think this is only related to the horizontal row

        if a is not None : # Nawid - a is an auxillary variable, I believe it is the output from vertical stack
            x += self.nin_skip(self.nonlinearity(a)) # Nawid - I beleive this is the addition of the skip connection - Performs concat elu non linearity to change the number of feature maps and then uses nin_skip to change the dimensions so that it matches the x variable - I think this uses the concat relu first to double the size of it and then uses nin_skip to make it so that the dimensions of the vertical stack output and the horizontal stack output are the same so they can add togheter

        x = self.nonlinearity(x) # Nawid- Performs concat elu non linearity to double its size
        x = self.dropout(x) # Nawid- Performs dropout
        x = self.conv_out(x) # Nawid - Performs convolution after obtaining information of the auxillary variable - Performs the output convolution which keeps its size the same
        if h is not None: # Nawid-  This is when the latent term is not zero
            h_shp = h.size()
            #print(h_shp)
            h = h.view(h_shp[0], h_shp[1],1,1) # Nawid - Change into format of (N,C,H,W) -
            x += self.hw(h) # Nawid - Could check for the case where there is a non-linearity here if that makes a difference - The shape should be fine as it should broadcast to the correct shape (similar as TF repo)
            #latent_term = self.hw(h).view(x.size()[0],1,1,2*num_filters)
            #x += latent_term #torch.mm(h,hw)

        a, b = torch.chunk(x, 2, dim=1) # Nawid - breaks up x into 2 chunks which corresponds to the part where it undergoes a tanh and a sigmoid ( tanh does not actually occur it seems, a seems to be the tanh term which it is multiplied by)
        c3 = a * F.sigmoid(b) # Nawid- obtains c3 which I am not sure how it is obtained - C3 is the multiplication of the tanh and the sigmoid output
        return og_x + c3
