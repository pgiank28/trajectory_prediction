from l5kit.data import LocalDataManager,ChunkedDataset
from l5kit.configs import load_config_data
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from torchvision import transforms
from torch import nn,optim
from math import ceil
from pytorch_pfn_extras.training import IgniteExtensionsManager
from pytorch_pfn_extras.training.triggers import MinValueTrigger
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset,Dataset
from ignite.engine import Engine
from typing import Callable

import pytorch_pfn_extras.training.extensions as E
import pytorch_pfn_extras as ppe
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
import l5kit
import yaml
import zarr
import os 


########################################################################################
##                                                                                    ##
##                      HELPERS                                                       ##
##                                                                                    ##
########################################################################################       
class AgentPointEmbedding(nn.Module):
    '''
    Spatial embedding for a single agent to be used as input in LSTM encoder.
    It is a MLP that takes as input the (x,y) tuple of agent's relative position and outputs a
    [1 x lstm_enc_input_dim] tensor for each agent
    '''
    def __init__(self,input_dim=2,hidden_layers_dims=32,lstm_enc_input_dim=256):
        super(AgentPointEmbedding,self).__init__()
        self._input_dim = input_dim
        self._output_dim = lstm_enc_input_dim
        self._embed_layer_hid = nn.Linear(input_dim,hidden_layers_dims)
        self._embed_layer_out = nn.Linear(hidden_layers_dims,lstm_enc_input_dim)
        
    def forward(self,positions):
        '''
        Inputs:
        - positions: Tensor of shape (agent_traj_len, batch size, 2)
        Output:
        - embedding: Tensor of shape (agent_traj_len, batch size, self.output_dim)
        '''
        
        batch_size = positions.size(1)
        embedding = self._embed_layer_hid(positions.view(-1,self._output_dim))
        embedding = self._embed_layer_out(embedding)
        return embedding.view(-1,batch_size,self._output_dim)
    
class AgentEncoderRNN(nn.Module):
    '''
    This part of the code is revised from MATF paper.
    Link to their original code: https://github.com/programmingLearner/MATF-architecture-details
    [
	 Presented at CVPR 2019]
    run on all the agents individually: batch_idx
    '''
    def __init__(self,hidden_dim=320,num_layers=50,input_dim=256,dropout=0.35):
        super(AgentEncoderRNN,self).__init__()
        
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._encoder = nn.LSTM(input_dim,hidden_dim,num_layers,dropout=dropout)
        
    def forward(self,trajectory):
        h0 = torch.zeros(self._num_layers, batch_size, self._h_dim).cuda()
        c0 = torch.zeros(self._num_layers, batch_size, self._h_dim).cuda()
        output,state = self._encoder(trajectory,(h0,c0))
        return state[0]
    
class SceneExtraction(nn.Module):
    '''
    The convolutinal neural network for segmentation extraction of the eye-bird-view of the scene.
    It is a pretrained,in Imagenet dataset, resnet34 network
    '''
    def __init__(self,output_shape):
        super(SceneExtraction,self).__init__()
        
        self.output_shape = output_shape
        ### Scene feature scene extraction by using 34-layer resnet where we have removed the last fully connected layer
        ### and added a output_shape=[C,H,W] final layer.
        resnet34 = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
        modules=list(resnet34.children())[:-1]
        
        self.model=nn.Sequential(*modules)
        
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.compress1 = nn.Conv2d(in_channels = 512, out_channels = self.output_shape[0],  kernel_size = (3,3), stride = 1, padding = 0)
        self.compress2 = nn.BatchNorm2d(100)
        
        
    def forward(self,image):
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        x = self.model(input_batch)
        
        ###  Fully Convolutional Networks for Semantic Segmentation by Jonathan Long, Evan Shelhamer, Trevor Darrell. ###
        x = F.interpolate(x,size=(self.output_shape[1],self.output_shape[2]))
        
        x = self.compress1(x)
        return nn.ReLU(self.compress2(x))
   

class AttentionTensorCreation(nn.Module):
    '''
    The concatentation of the scene extraction tensor and the multi-agent LSTM encoding tensor.
    The spatial structure of the agents in the scene(at the last frame)is retained.
    This part of code is from Multi-agent Tensor Fusion paper
    '''
    def __init__(self):
        super(AttentionTensorCreation,self).__init__()
        
    def forward(self,input_grid,input_state_of_agent,coordinates_at_last_frame,scene_id):
        '''
        Params: input_grid: spatial grid input of shape (batch * c * h * w), 
                input_state_of_agent: input state vector (output from agent encoder) of shape
                            (batch (must = 1) * c * 1)
                coordinates_at_last_frame: input coordinate of shape (2)
                scene_id: int, for scene, not for agent
        Returns: placed and pooled map of agents
        '''
        
        ori_state = input_grid[scene_id, :, coordinate[0], coordinate[1]]
        pooled_state = torch.max(ori_state.type(torch.cuda.FloatTensor), input_state_of_agent[0, :, 0].type(torch.cuda.FloatTensor))

        input_grid[scene_id, :,coordinate[0], coordinate[1]] = pooled_state
        return input_grid
     
        
class AttentionTensorFusion(nn.Module):
    '''
    The convolution neural network that takes as input the multiAgent tensor and outputs the attention tensor for the GAN. 
    From MATF paper
    '''
    def __init__(self,input_channels,output_channels):
        
        super(AttentionTensorFusion,self).__init__()
        
        self._conv1 = conv2DBatchNormRelu(input_channels, output_channels, k_size = 3, stride = 1, padding = 1, dilation = 1)
        self._pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        
        self._conv2 = conv2DBatchNormRelu(output_channels, output_channels, k_size = 3,  stride = 1, padding = 1, dilation = 1)
        self._pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        
        self._conv3 = conv2DBatchNormRelu(output_channels, output_channels, k_size = 4,  stride = 1, padding = 1, dilation = 1)
        
        self._deconv2 = deconv2DBatchNormRelu(output_channels, output_channels, k_size = 4, stride = 2, padding = 1)

    def forward(self,encoded_image,encoded_agent_traj):
        cat = torch.cat((encoded_image.type(torch.cuda.FloatTensor), encoded_agent_traj.type(torch.cuda.FloatTensor)), 1)
        
        conv1 = self._conv1.forward(cat)
        conv2 = self._conv2.forward(self._pool1.forward(conv1))
        conv3 = self._conv3.forward(self._pool2.forward(conv2))

        up2 = self._deconv2.forward(conv2)
        up3 = F.upsample(conv3, scale_factor=5)

        features = conv1 + up2 + up3
        return features

class AgentDecoderRNN(nn.Module):
    '''
    The LSTM RNN decoder,that makes the predictions of the future paths.
    It is the generator of the GAN.
    '''
    def __init__(self,noise,AttentionTensorChannelSize=4096,sequence_length=50,num_layers=1,dropout=0.0):
        super(AgentDecoderRNN,self).__init__()
        self._decoder = nn.LSTM(AttentionTensorChannelSize,AttentionTensorChannelSize,num_layers,dropout)
        self._attention_weights = torch.rand(1,AttentionTensorChannelSize)
        self._seq_len = sequence_length
        self._white_noise = noise
        
    def forward(self,AttentionTensor):
        '''
        Initialize the generator
        '''
        state = self._white_noise
        weighted_input = AttentionTensor
        states = []
        '''
        Generate predictions
        '''
        for _ in range(self._seq_len):
            output, state = self._decoder(weighted_input,state)
            weighted_input = torch.mul(state,self._attention_weights)
            weighted_input = torch.mul(AttentionTensor,weighted_input)#Be careful
            states.append(state)
            
        return torch.FloatTensor(states)
    
class Discriminator(nn.Module):
    '''
    The classifier that clasifies the predicted paths as fake or real
    '''
    def __init__(self,embed_dimension,hidden_layer, dropout):
        super(Discriminator, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(embed_dimension, hidden_layer),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_layer, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self._classifier(x)
    
