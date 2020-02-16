import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalEncoder(nn.Module):
    def __init__(self,in_channels,latent_dim,size):
        super(ConvolutionalEncoder,self).__init__()
        self.size = size
        self.scaling = self.size//32
        self.conv1 = nn.Conv2d(in_channels,64,4,stride=2, padding=1)
        self.conv2 = nn.Conv2d(64,128,4,stride=2, padding=1)
        self.conv3 = nn.Conv2d(128,256,4,stride = 2, padding=1)
        self.fc1 = nn.Linear(self.scaling*4*self.scaling*4*256,512)
        self.mean = nn.Linear(512,latent_dim)
        self.std = nn.Linear(512,latent_dim)

    def encode(self,x): # Nawid - Performs the amortised inference where a mean and variance of a gaussian is obtained based on the input x
        #print('first x', x.size())
        x = F.leaky_relu(self.conv1(x))

        #print('conv1 output',x.size())
        x = F.leaky_relu(self.conv2(x))

        #print('conv2 output',x.size())
        x = F.leaky_relu(self.conv3(x))
        #print('conv3 output',x.size())
        x = x.view(-1,self.scaling*4*self.scaling*4*256)
        #print('conv1 output x', x.size())
        x = self.fc1(x)
        #print('fc1 output',x.size())
        mean = self.mean(x)
        #print('mean output',mean.size())
        std = F.sigmoid(self.std(x))
        #print('std output',std.size())
        #std = torch.max()
        return mean, std

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self,x):
        mu, logvar = self.encode(x) # Nawid - Obtain values for the mean and the log variance
        z = self.reparameterize(mu,logvar) # Nawid - Obtain a stochastic latent variable from the mean and the variance term from the network
        #print('z output', z.size())
        return z,mu, logvar # Nawid - Outputs mu and logvar in order to calculate the ELBO_loss_term
