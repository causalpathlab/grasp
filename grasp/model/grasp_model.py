import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random 
import logging
logger = logging.getLogger(__name__)


		  
class Stacklayers(nn.Module):
	"""
	Stacklayers
  
    Parameters
    ----------
	input_size: dimension of the input vector
	layers: list with hidden layer sizes
	dropout: proportion for dropout

	"""
	def __init__(self,input_size,layers,dropout=0.1):
		super(Stacklayers, self).__init__()
		self.layers = nn.ModuleList()
		self.input_size = input_size
		for next_l in layers:
			self.layers.append(nn.Linear(self.input_size,next_l))
			self.layers.append(nn.BatchNorm1d(next_l))
			self.layers.append(self.get_activation())
			self.layers.append(nn.Dropout(dropout))
			self.input_size = next_l

	def forward(self, input_data):
		for layer in self.layers:
			input_data = layer(input_data)
		return input_data

	def get_activation(self):
		return nn.ReLU()

class MLP(nn.Module):
    def __init__(self,
        input_dims:int,
        layers:list
        ):
        super(MLP, self).__init__()
        
        self.fc = Stacklayers(input_dims,layers)

    def forward(self, x:torch.tensor):
        z = self.fc(x)
        return z

###### GRASP BASE MODEL #######

class GRASPBaseNet(nn.Module):
	def __init__(self,input_dim,latent_dim,enc_layers,dec_layers,num_batches):
		super(GRASPBaseNet,self).__init__()
  
		self.u_encoder = MLP(input_dim,enc_layers)
		self.u_decoder = MLP(latent_dim,dec_layers)
  
		decoder_in_dim = dec_layers[len(dec_layers)-1]  
		self.zinb_scale = nn.Linear(decoder_in_dim, input_dim) 
		self.zinb_dropout = nn.Linear(decoder_in_dim, input_dim)
		self.zinb_dispersion = nn.Parameter(torch.randn(input_dim), requires_grad=True)
		
		self.batch_discriminator = nn.Linear(latent_dim, num_batches)

	
	def forward(self,x_c1):	
 
		row_sums = x_c1.sum(dim=1, keepdim=True)
		x_norm = torch.div(x_c1, row_sums) * 1e4
  
		z = self.u_encoder(x_norm.float())
		h = self.u_decoder(z)
  
		px_scale = torch.exp(self.zinb_scale(h))  
		px_dropout = self.zinb_dropout(h)  
		px_rate = self.zinb_dispersion.exp()
  
		batch_pred = self.batch_discriminator(z)
		
		return z,px_scale,px_rate,px_dropout,batch_pred


###### GRASP UNIQUE MODEL #######

class GRASPUniqueNet(nn.Module):
	def __init__(self,input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches):
		super(GRASPUniqueNet,self).__init__()
		self.u_encoder = MLP(input_dim,enc_layers)

		concat_dim = common_latent_dim + unique_latent_dim 
		self.u_decoder = MLP(concat_dim,dec_layers)
  
		decoder_in_dim = dec_layers[len(dec_layers)-1]  
		self.zinb_scale = nn.Linear(decoder_in_dim, input_dim) 
		self.zinb_dropout = nn.Linear(decoder_in_dim, input_dim)
		self.zinb_dispersion = nn.Parameter(torch.randn(input_dim), requires_grad=True)
		
		self.batch_discriminator = nn.Linear(unique_latent_dim, num_batches)

	
	def forward(self,x_c1,x_zcommon):	
 
		row_sums = x_c1.sum(dim=1, keepdim=True)
		x_norm = torch.div(x_c1, row_sums) * 1e4
  
		z_unique = self.u_encoder(x_norm.float())
		
		z_combined = torch.cat((x_zcommon, z_unique), dim=1)

		h = self.u_decoder(z_combined)
  
		px_scale = torch.exp(self.zinb_scale(h))  
		px_dropout = self.zinb_dropout(h)  
		px_rate = self.zinb_dispersion.exp()
  
		batch_pred = self.batch_discriminator(z_unique)
		
		return z_unique,px_scale,px_rate,px_dropout,batch_pred


###### GRASP UNIQUE GRAPH MODEL #######

from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import SAGEConv

class GRASPUniqueGNET(nn.Module):
	def __init__(self,input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches):
		super(GRASPUniqueGNET,self).__init__()
		self.u_encoder = Stacklayers(input_dim,enc_layers)

		self.u_gnn = SAGEConv(unique_latent_dim , unique_latent_dim )
  
		self.w_common = nn.Linear(common_latent_dim, unique_latent_dim, bias=False)
  
		decoder_in_dim = common_latent_dim + unique_latent_dim 
		self.zinb_scale = nn.Linear(decoder_in_dim, input_dim) 
		self.zinb_dropout = nn.Linear(decoder_in_dim, input_dim)
		self.zinb_dispersion = nn.Parameter(torch.randn(input_dim), requires_grad=True)
		
		self.batch_discriminator = nn.Linear(unique_latent_dim, num_batches)

	
	def forward(self,x_c1,x_zcommon,edge_index):	
 
		row_sums = x_c1.sum(dim=1, keepdim=True)
		x_norm = torch.div(x_c1, row_sums) * 1e4
  
		z_unique = self.u_encoder(x_norm.float())

		z_unique = self.u_gnn(z_unique, edge_index)
		
		z_common_proj = self.w_common(x_zcommon)
		z_unique = F.relu(z_unique - z_common_proj)
  
		h = torch.cat((x_zcommon, z_unique), dim=1)
  
		px_scale = torch.exp(self.zinb_scale(h))  
		px_dropout = self.zinb_dropout(h)  
		px_rate = self.zinb_dispersion.exp()
  
		batch_pred = self.batch_discriminator(z_unique)
		
		return z_unique,px_scale,px_rate,px_dropout,batch_pred
