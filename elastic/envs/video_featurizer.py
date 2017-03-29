'''Outer AutoEncoder. This file contains the implementation of an autoencoder. 
Versions available:
1. Multilayer perceptron
'''
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os.path
from scipy.misc import imresize

EPSILON = 1e-6

# Requirements : Need torch with cuda enabled
'''
config_save_file = 'save_file.pickle'
featurizer = ConvolutionalAutoEncoder(config_save_file)
goals, variances = featurizer.getGoals_and_Variances()
embedding = featurizer.get_latent(obs) #obs is assumed to be 5 windowed frames stacked along the color channel
'''

class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, config_file):
        """init method for ConvolutionalAutoEncoder
        param array encoder_params: an array of tuples defining the parameters of the encoder. See http://pytorch.org/docs/nn.html for the format of the tuples. The tuples must be in the format (input_channels, output_channels, kernel_size, stride, padding). 
        for conv and convTranspose  layers and (kernel_size, stride, padding) for max(un)pool layers. Padding and Kernel sizes are tuples. 
        encoder_params[0][0] must be equal to the number of input channels in the image. See the network structure for the number of layers. Len(encoder_params) must equal to 6
        param dropout: the fraction of neurons to dropout in the dropout layers
        """

        super(ConvolutionalAutoEncoder, self).__init__()
        with open(config_file, 'rb') as input_file:
            self.config_map = pickle.load(input_file, encoding='latin1')
        
        encoder_params = self.config_map['embedding_dims']
        latent_fc = self.config_map['encoder_fc_dims']
        state_dict = self.config_map['state_dict']
        self.config_map['use_gpu'] = False

        state_augmentations = self.config_map['state_augmentation_embeddings']
        self.goal_states = [mu_var[0] for mu_var in state_augmentations]
        self.variances = [mu_var[1] for mu_var in state_augmentations]

        self.fconv1 = nn.Conv2d(*encoder_params[0])
        self.fmaxpool1 = nn.MaxPool2d(*encoder_params[1], return_indices=True)
        self.fconv2 = nn.Conv2d(*encoder_params[2])

        self.rconvtrans1 = nn.ConvTranspose2d(*encoder_params[3])
        self.rmaxunpool2 = nn.MaxUnpool2d(*encoder_params[4])
        self.rconvtrans2 = nn.ConvTranspose2d(*encoder_params[5])
        print('latent_fc', latent_fc)
        self.fc = nn.Linear(*latent_fc)
        self.load_state_dict(state_dict)

    def get_latent(self, x):
        '''Gets the embedding of x by passing x through the encoder layers 
        param tensor x: The tensor to Autoencode.
        ''' 
        # We assume input is a list of frame of Txchannelxwidthxheight
        # If we get widthxheight, we convert to 1xchannelxwidthxheight
        # Assumes numpy
        if len(x.shape) == 3:
            x = np.array([x])
        x_dims = np.shape(x)
        new_x = np.zeros(shape=x_dims[:3] + (120,))
        # Convert each image to 160x120
        for i in range(len(x)):
            image = x[i, :, :, :].astype(float)
            image = image[:, :, 40:].T # For each frame, we cut off the top forty
            image = imresize(image, (120, 160, 3))
            new_x[i, :, :, :] = image.T # transpose to restore appropriate dimensions
            new_x[i, :, :, :] -= np.mean(new_x[i, :, :, :])
            new_x[i, :, :, :] /= (np.std(new_x[i, :, :, :]) + EPSILON)
        shape = new_x.shape
        new_x = np.reshape(new_x, (1, shape[0]*shape[1], shape[2], shape[3]))
        # print new_x.shape
        # Convert to pytorch tensor now
        torch_x = torch.from_numpy(new_x).float()
        # Convert to Pytorch variables too
        var_x = Variable(torch_x.cuda()) if self.config_map['use_gpu'] else Variable(torch_x)
        
        # Now get latent
        hidden = self.fconv1(var_x) 
        hidden = F.relu(hidden)
        hidden, _ = self.fmaxpool1(hidden)
        hidden = self.fconv2(hidden)
        hidden = F.relu(hidden)
        hidden = hidden.view(-1, int(np.prod(hidden.data.size()[1:])))
        hidden = F.relu(self.fc(hidden))
        return torch.Tensor.numpy(hidden.data) if not self.config_map['use_gpu'] else hidden.data.cpu().numpy()

    def get_goals_and_variances(self):
        return self.goal_states, self.variances

