import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import  minimal_overlap_loss_pair, get_zinb_reconstruction_loss,minimal_overlap_loss_triplet
from torch_geometric.loader import RandomNodeLoader,NeighborLoader
import logging
logger = logging.getLogger(__name__)
import numpy as np
import random 
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from torch_geometric.data import ClusterData, ClusterLoader
from itertools import cycle


def grasp_train_base(model,data,l_rate,epochs=100):

	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	epoch_losses = []
	criterion = nn.CrossEntropyLoss() 
	for epoch in range(epochs):
		epoch_l,el_recon,el_batch = (0,)*3
		for x_c1,y,batch in data:
			opt.zero_grad()
			z,px_s,px_r,px_d,batch_pred = model(x_c1)
			train_loss_recon = get_zinb_reconstruction_loss(x_c1,px_s, px_r, px_d)
			train_loss_batch = criterion(batch_pred, batch)
			train_loss = train_loss_recon + train_loss_batch
			train_loss.backward()

			opt.step()
			epoch_l += train_loss.item()
			el_recon += train_loss_recon.item()
			el_batch += train_loss_batch.item()
		   
		epoch_losses.append([epoch_l/len(data),0.0,el_recon/len(data),el_batch/len(data)])  
		
		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch,epoch_l/len(data) ))

	return epoch_losses

def grasp_train_unique_gnn(model,dataset,l_rate,epochs,device):

	logger.info('Init training....nn_unq_graph')
	model.train()
	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	epoch_losses = []
	criterion = nn.CrossEntropyLoss() 
	
	x_be_graph = dataset.x_be_data
 
	x_be_data = ClusterData(x_be_graph, num_parts=10, recursive=False)
	x_be_loader = ClusterLoader(x_be_data, batch_size=1, shuffle=True,num_workers=1)
 
	
	epoch_losses = []
	for epoch in range(epochs):
		epoch_l, el_z, el_recon, el_batch, el_group = 0, 0, 0, 0,0
  
		for x_be_batch in x_be_loader:
	  
			x_c1 = x_be_batch.x.to(device)
			x_be_edge_index = x_be_batch.edge_index.to(device)
			x_indxs = x_be_batch.y

			batch_labels = dataset.x_batch_data.x[x_indxs].to(device)
			group_labels = dataset.x_group_data.x[x_indxs].to(device)
   
			mask_0 = torch.isin(dataset.x_ge_data.x[0],x_indxs)
			mask_1 = torch.isin(dataset.x_ge_data.x[1],x_indxs)
			mask = mask_0 & mask_1
   
			x_ge_edge_index = dataset.x_ge_data.x[:,mask]
			index_map = {value.item(): idx for idx, value in enumerate(x_indxs)}
			x_ge_edge_index_be_space = torch.tensor([[index_map[node.item()] for node in edge] for edge in x_ge_edge_index])
			x_ge_edge_index_be_space = x_ge_edge_index_be_space.to(device)
   
			opt.zero_grad()
   
			z_mix,z_be, z_ge, z_un, px_s, px_r, px_d, batch_pred,group_pred = model(x_c1, x_be_edge_index, x_ge_edge_index_be_space)
   
			train_loss_z = minimal_overlap_loss_triplet(z_be,z_ge,z_un)
			train_loss_recon = get_zinb_reconstruction_loss(x_c1,px_s, px_r, px_d)
			train_loss_batch = criterion(batch_pred, batch_labels)
			train_loss_group = criterion(group_pred, group_labels)
   
			train_loss = train_loss_z + train_loss_recon + train_loss_batch + train_loss_group
			train_loss.backward()

			opt.step()
			epoch_l += train_loss.item()
			el_z += train_loss_z.item()
			el_recon += train_loss_recon.item()
			el_batch += train_loss_batch.item()
			el_group += train_loss_group.item()
		   
		epoch_losses.append([epoch_l/len(dataset),el_z/len(dataset),el_recon/len(dataset),el_batch/len(dataset),el_group/len(dataset)])  
		
		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch,epoch_l/len(dataset) ))

	return epoch_losses
 
 