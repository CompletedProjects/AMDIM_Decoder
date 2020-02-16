import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_residual(channels): # Nawid- Performs a 3x3 convolution followed by a 1x1 convolution - The 3x3 convolution is padded and so the overall shape is the same.
    return nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels, channels, 1),
    )

class HalfEncoder(nn.Module):
    """
    An encoder that cuts the input size in half in both
    dimensions.
    """
    def __init__(self, in_channels, out_channels):
        super(HalfEncoder,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,3, stride=2, padding=1)
        self.residual1 = _make_residual(out_channels)
        self.residual2 = _make_residual(out_channels)

    def forward(self,x):
        x = self.conv(x)
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        return x

class HalfQuarterDecoder(nn.Module):
    """
    A decoder that takes two inputs. The first one is
    upsampled by a factor of two, and then combined with
    the second input which is further upsampled by a
    factor of four.
    """
    def __init__(self,in_channels, out_channels):
        super(HalfQuarterDecoder, self).__init__()
        self.residual1 = _make_residual(in_channels)
        self.residual2 = _make_residual(in_channels)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.residual3 = _make_residual(in_channels)
        self.residual4 = _make_residual(in_channels)
        self.conv3 = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self,inputs):
        assert len(inputs) == 2
        # Upsample the top input to match the shape of the
        # bottom input.
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = F.relu(x)
        x = self.conv1(x) # Nawid - This is a convolution transpose which make the top input match the shape of the bottom input
        x = F.relu(x)

        # Mix together the bottom and top inputs.
        x = torch.cat([x, inputs[1]], dim=1) # Nawid - Concatentate the upsample top feature map with the bottom feature map
        x = self.conv2(x) # Nawid - Downsamples

        x = x + self.residual3(x)
        x = x + self.residual4(x)
        x = F.relu(x)
        x = self.conv3(x) # Nawid - Upsamples
        x = F.relu(x)
        x = self.conv4(x) # Nawid - Upsamples
        return x

class VQ_VAE_Decoder(nn.Module):
    '''
    Performs the encoder to get the second input and then performs the decoder with both of the different inputs
    '''
    def __init__(self,in_channels,intermediate_channels, out_channels):
        super(VQ_VAE_Decoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, intermediate_channels,3, stride=1, padding=1)
        self.residual1 = _make_residual(intermediate_channels)
        self.encoder =HalfEncoder(intermediate_channels,intermediate_channels)
        self.halfquarterdecoder = HalfQuarterDecoder(intermediate_channels, out_channels)

    def forward(self,x):
        half_quarter_inputs = []
        x = x.view(-1, 4, 16, 16)
        x = self.conv(x) # Nawid - This gives c x 16 x 16
        #print('conv1',x.size())
        x =  self.residual1(x) # Nawid - This should give c x 16 x 16
        #print('first residual',x.size())
        half_x = self.encoder(x) # Nawid - This should give c x 8 x 8
        #print('encoded shape',half_x.size())
        half_quarter_inputs.append(half_x) # Nawid -  half_x into the network
        half_quarter_inputs.append(x) # Nawid - Places x into the inputs

        reconstruction = self.halfquarterdecoder(half_quarter_inputs)
        return reconstruction
