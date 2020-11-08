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
    


##################################################################################################################
##                                                                                                              ##
##                                          MODEL                                                               ##
##                                                                                                              ##
##################################################################################################################
class MultitensorAttentionModel(nn.Module):
    def __init__(self,point_dim,LSTM_encoder_input_size,LSTM_encoder_hidden_size,LSTM_size,CNN_output_shape,
                 AT_channel_size,LSTM_decoder_seq_len,GAN_noise,LSTM_decoder_layers_per_sequence,
                 MATF_RNN_layer_size,Generator_prediction_length,GAN_discriminator_hidden_size,groundtruth_paths):
        
        super(MultitensorAttentionModel,self).__init__()
        self._agent_point_embedding = AgentPointEmbedding(input_dim=point_dim,hidden_layers_dims=32,lstm_enc_input_dim=LSTM_encoder_input_size)
        self._agent_encoder_RNN = AgentEncoderRNN(hidden_dim=LSTM_encoder_hidden_size,num_layers=LSTM_size,input_dim=LSTM_encoder_input_size,dropout=0.35)
        self._scene_encoder_CNN = SceneExtraction(CNN_output_shape)
        self._concatTensors = AttentionTensorCreation()
        self._fuseConcatTensors = AttentionTensorFusion(CNN_output_shape[0]*2,MATF_RNN_layer_size)
        self._generator = AgentDecoderRNN(GAN_noise,AT_channel_size,LSTM_decoder_seq_len,LSTM_decoder_layers_per_sequence,0.0)
        self._discriminator = Discriminator(Generator_prediction_length,GAN_discriminator_hidden_size,0.0)
        
    #def forward(self,agent_trajectories,scene,scene_id,agents_coord):
        
    def forward(self,scene):
        
        #Past agent's trajectories
        x = scene['past_list']
        #x = x.permute(1,0,2)
        print(x.shape)
        #x = self._agent_point_embedding(x)
        #x = self._agent_encoder_RNN(x)
        '''
        x2 = scene['scene_image']
        x2 = list2batch(x2)
        x2 = self._scene_encoder_CNN(x2)
        
        agents_in_scene = scene['num_agents']
        
        input_grid = torch.zeros(agents_in_scene,x2.shape,device='cuda')
        
        agents_coord = list2batch(scene['coord_list'])
        
        scene_id = scene['scene_id']
        
        for agent in range(agents_in_scene):
            input_grid = self._concatTensors(input_grid,x.view(1,x2.shape[0],1),agents_coord[agent],scene_id)
            
        x = self._fuseConcatTensors(x2,input_grid)
        
        x = self._generator(x)
        
        x = torch.cat((x,groundtruth_paths),dim=0)
        x = x[torch.randperm(x.size()[0])]
        return self._discriminator(x)'''
        return scene**2
 
##################################################################################################################
##                                                                                                              ##
##                                           UTILITIES                                                          ##
##                                                                                                              ##
##################################################################################################################
class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),) 

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    
class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs

def list2batch(seq):
        # assemble a list of elements to batch, batch_idx: 0th dimension
        stacked = torch.tensor(seq[0]).unsqueeze(0)
        i = 1
        l = len(seq)
        while i < l:
            stacked = torch.cat((stacked, torch.tensor(seq[i]).unsqueeze(0)), 0)
            i += 1
        return stacked

def batch2list(batch):
        # dis-assemble batch (index 0) to a list of elements
        unstacked = torch.unbind(batch, 0)
        return unstacked
    
def save_yaml(filepath, content, width=120):
    with open(filepath, 'w') as f:
        yaml.dump(content, f, width=width)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        content = yaml.safe_load(f)
    return content


class DotDict(dict):
    """dot.notation access to dictionary attributes

    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
        

##################################################################################################################
##                                                                                                              ##
##                                           TRAINING                                                           ##
##                                                                                                              ##
##################################################################################################################
def create_trainer(model, optimizer, device) -> Engine:
    model.to(device)

    def update_fn(engine, batch):
        model.train()
        optimizer.zero_grad()
        loss, metrics = model(*[elem.to(device) for elem in batch])
        loss.backward()
        optimizer.step()
        return metrics
    trainer = Engine(update_fn)
    return trainer

class TransformDataset(Dataset):
    def __init__(self,pytorch_dataset,length):
        self.dataset = pytorch_dataset
        self.len = length
        
    def __getitem__(self,index):
        batch = self.dataset.get_scene_indices(index)
        past = []
        coords = []
        for i,x in enumerate(batch):
            past.append(self.dataset[x]["target_positions"])
            coords.append(self.dataset[x]["target_positions"][-1])
           
        
        result= {'scene_id':index,
                 'scene_image':self.dataset[0]["image"],
                 #'past_list':list2batch(past),
                 #'coord_list':list2batch(coords),
                 'num_agents':len(batch)}
        
        return result
    
    def __len__(self):
        return self.len


conf = {   'model_params':{
                        'point_dim':2,
                        'D':128,
                        'H':256,
                        'T':99,
                        'CNN_shape':[32,64,64],
                        'Attention_columns':4096,
                        'future_num_frames':50,
                        'Attention_channels':32,
                        'Classifier_layers':16
                },
            'device':'cuda',
            'debug':True,
            'epoch':1,
            'out_dir':'results/train',
            'snapshot_freq':50
}

GAN_noise = torch.randn(conf['model_params']['Attention_channels']) ##size:[Attention_channels,1]
groundtruth_paths = torch.randn(1000,conf['model_params']['future_num_frames'],2)##size:[N,future_num_frames,2]

model = MultitensorAttentionModel(conf['model_params']['point_dim'],conf['model_params']['D'],conf['model_params']['H'],conf['model_params']['T'],conf['model_params']['CNN_shape'],
                 conf['model_params']['Attention_columns'],conf['model_params']['future_num_frames'],GAN_noise,1,
                 conf['model_params']['Attention_channels'],conf['model_params']['future_num_frames'],conf['model_params']['Classifier_layers'],groundtruth_paths)

optimizer = optim.Adam(model.parameters(),lr=1e-3)

trainer = create_trainer(model,optimizer,conf['device'])

print("Model created")

os.environ["L5KIT_DATA_FOLDER"] = '/kaggle/input/lyft-motion-prediction-autonomous-vehicles'
cfg = load_config_data("/kaggle/input/lyft-config-files/visualisation_config.yaml")

dm = LocalDataManager()
dataset_path = dm.require('scenes/sample.zarr')
validataset_path = dm.require('scenes/validate.zarr')

zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
val_chunked_dataset = ChunkedDataset(validataset_path)
val_chunked_dataset.open()

cfg["raster_params"]["map_type"] = "py_semantic"
semantic_rasterizer = build_rasterizer(cfg,dm)
'''
#Load PyTorch datasets
'''
Ego_dataset = EgoDataset(cfg,zarr_dataset,semantic_rasterizer)
Agent_Dataset = AgentDataset(cfg,zarr_dataset,semantic_rasterizer)
Agent_Valid_Dataset = AgentDataset(cfg,val_chunked_dataset,semantic_rasterizer)


agent_final_dataset = TransformDataset(Agent_Dataset,length=4)#170000 scenes
agent_valid_dataset = TransformDataset(Agent_Valid_Dataset,length=4)



#Scene by scene dataloader
Agent_Final_Dataloader = DataLoader(agent_final_dataset,shuffle=False,batch_size=16,num_workers=4) 

validation_loader = DataLoader(agent_valid_dataset,shuffle=False,batch_size=16,num_workers=4)
print("Datasets ready")


def eval_func(*batch):
    loss, metrics = model(*[elem.to(device) for elem in batch])


valid_evaluator = E.Evaluator(
    validation_loader,
    model,
    progress_bar=False,
    eval_func=eval_func,
)

log_trigger = (10 if conf["debug"] else 1000, "iteration")
log_report = E.LogReport(trigger=log_trigger)


extensions = [
    log_report,  # Save `log` to file
    valid_evaluator,  # Run evaluation for valid dataset in each epoch.
    # E.FailOnNonNumber()  # Stop training when nan is detected.
    E.ProgressBarNotebook(update_interval=10 if conf["debug"] else 100),  # Show progress bar during training
    E.PrintReportNotebook(),  # Show "log" on jupyter notebook  
]

epoch = conf["epoch"]

models = {"main": model}
optimizers = {"main": optimizer}
manager = IgniteExtensionsManager(
    trainer,
    models,
    optimizers,
    epoch,
    extensions=extensions,
    out_dir=str(conf["out_dir"]),
)
# Save predictor.pt every epoch
manager.extend(E.snapshot_object(model, "predictor.pt"),
               trigger=(conf["snapshot_freq"], "iteration"))
# Check & Save best validation predictor.pt every epoch
# manager.extend(E.snapshot_object(predictor, "best_predictor.pt"),
#                trigger=MinValueTrigger("validation/main/nll", trigger=(flags.snapshot_freq, "iteration")))
# --- lr scheduler ---
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-10)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99999)

manager.extend(lambda manager: scheduler.step(), trigger=(1, "iteration"))
# Show "lr" column in log
manager.extend(E.observe_lr(optimizer=optimizer), trigger=log_trigger)

# --- HACKING to fix ProgressBarNotebook bug ---
manager.iteration = 0
manager._iters_per_epoch = len(Agent_Final_Dataloader)
print("Training starting")
trainer.run(Agent_Final_Dataloader, max_epochs=epoch)
