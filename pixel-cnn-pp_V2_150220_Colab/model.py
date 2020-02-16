import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from utils import *
import numpy as np
from encoder import ConvolutionalEncoder

class PixelCNNLayer_up(nn.Module): # Nawid - This forms the resnet block when you move from the top of the image and you move downwards - THIS MAY REPRESNT THE UPSAMPLING LAYERS
    def __init__(self, nr_resnet, nr_filters,latent_dimension, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above - Nawid - This corresponds to the vertical stack
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,latent_dimension,
                                        resnet_nonlinearity, skip_connection=0) # Nawid -there are no skip connections in the vertical stack
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left - Nawid - This corresponds to the horizontal stack as shown by the skip connection being presented
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,latent_dimension,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)]) # Nawid - Define two different resnets, one for pixels above and one for pixels above and to the left

    def forward(self, u, ul, latent_vector= None): # Nawid - Set the latent vector to have a default of None to check if it works
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, h = latent_vector) # Nawid - u is the output of the vertical stack
            ul = self.ul_stream[i](ul, a=u, h = latent_vector) # Nawid - ul is the output from the horizontal stack which is dependent on the previous output which is ul and the auxillary variable is the output of the vertical stack
            u_list  += [u] # Nawid -list of outputs of the vertical stacks
            ul_list += [ul] # Nawid - List of outputs of the horizontal stack

        return u_list, ul_list

class PixelCNNLayer_down(nn.Module): # NAWID- THIS REPRESENTS THE DOWNSAMPLING LAYERS
    def __init__(self, nr_resnet, nr_filters,latent_dimension, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,latent_dimension,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)]) # Nawid- Build several resents

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,latent_dimension,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list, latent_vector= None):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop(), h = latent_vector) # Nawid - Uses the u as well as the u from the last entry of u_list
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1), h = latent_vector) # Nawid - Uses ul as well as concatenated ul_list and u

        return u, ul


class PixelCNN(nn.Module): # Nawid - nr_resnet = 5 means that there are 5 resnet layers in each block ( which was in the pixelCNN++ paper)
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3, image_size = 32,latent_dimension=20): # Nawid - number of input channels of the image and I added the latent dimension for the decoder
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' :
            self.resnet_nonlinearity = lambda x : concat_elu(x) # Nawid - Non linearity is concat
        else :
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.image_size = image_size # Nawid - Hardcoding a different size at the moment.

        self.latent_dimension = latent_dimension
        self.encoder = ConvolutionalEncoder(input_channels,latent_dimension,self.image_size) # Nawid - Instantiates the Convolutional Encoder

        self.nr_filters = nr_filters # Nawid -  Number of filters
        self.input_channels = input_channels # Nawid - Number of channels fo the image
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0)) # Nawid - Right padding
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0)) # Nawid - Downwards padding


        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2 # Nawid - I believe this will be a list where there is one input which has nr_resnet and two other inputs which are nr_resnet+1 large
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,latent_dimension,
                                                self.resnet_nonlinearity) for i in range(3)]) # Nawid-  Number of resnet blocks, there are 3 different resnet blocks where one block has nr_resnet number of layers whilst the other two blocks have nr_resnet+1 number of layers

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,latent_dimension,
                                                self.resnet_nonlinearity) for _ in range(3)]) # Nawod - 3 blocks of up sampling resnet blocks each with nr_resnet number of layers

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)]) # Nawid - This is used to downsample the stream from betweeen each resnet block, there are 2 different ones as they downsample between each resnet block (eg since there are 3 resnet blocks, there are 2 downsample which occurs between blocks)

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])
# Nawid - Downsamples the horizontal stack
        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])
# Nawid - Upsamples the vertical stack, there are 2 upsampling since there are 3 decoding Resnet blocks
        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])


        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3),
                        shift_output_down=True) # Nawid - Initial convolution

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)]) # Nawid-  performs convolution for the horizontal stack

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix) # Nawid - Network in network 1x1 convolutions to make the size between auxiallry variables the same
        self.init_padding = None


    def forward(self, x, sample=False):
        # similar as done in the tf repo :
        latent_vector, mu,logvar = self.encoder(x) # Nawid - Computes the latent_vector for the network


        if self.init_padding is None and not sample:
            xs = [int(y) for y in x.size()] # Nawid -  shape
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False) # Nawid  add channel of ones to distinguish image from padding later on
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample :
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False) # Nawid  add channel of ones to distinguish image from padding later on
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1) # Nawid - Concatenate the padding with x



        ###      UP PASS    ###
        # Nawid - This part is the initial causal convolution
        x = x if sample else torch.cat((x, self.init_padding), 1) # Nawid - It seems that padding is always added regardless
        u_list  = [self.u_init(x)] # Nawid - Initial convolution of the vertical stack
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)] # Nawid - Initial convolution of the horizontal stack, it performs two different convolutions which is generally due to horizontal stack taking both the vertical stack and the vertical stack information

        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1], latent_vector) # Nawid- Uses the output of the of the previous resnet block and passes it through the resent block
            u_list  += u_out
            ul_list += ul_out

            if i != 2: # Nawid - Downscale only the first two times
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])] # Nawid - Downsamples the last output
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list, latent_vector)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul)) # Nawid - Makes it so the output has the correct size

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out,mu,logvar # Nawid- I added mu and logvar for the regularisation


if __name__ == '__main__':
    ''' testing loss with tf version '''
    np.random.seed(1)
    xx_t = (np.random.rand(15, 32, 32, 100) * 3).astype('float32')
    yy_t  = np.random.uniform(-1, 1, size=(15, 32, 32, 3)).astype('float32')
    x_t = Variable(torch.from_numpy(xx_t)).cuda()
    y_t = Variable(torch.from_numpy(yy_t)).cuda()
    loss = discretized_mix_logistic_loss(y_t, x_t)

    ''' testing model and deconv dimensions '''
    x = torch.cuda.FloatTensor(32, 3, 32, 32).uniform_(-1., 1.)
    xv = Variable(x).cpu()
    ds = down_shifted_deconv2d(3, 40, stride=(2,2))
    x_v = Variable(x)

    ''' testing loss compatibility '''
    model = PixelCNN(nr_resnet=3, nr_filters=100, input_channels=x.size(1))
    model = model.cuda()
    out = model(x_v)
    loss = discretized_mix_logistic_loss(x_v, out)
    print('loss : %s' % loss.data[0])
