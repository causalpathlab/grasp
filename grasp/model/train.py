import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import  minimal_overlap_loss, get_zinb_reconstruction_loss
from torch_geometric.loader import RandomNodeLoader
import logging
logger = logging.getLogger(__name__)
import numpy as np
import random 
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


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
 
def grasp_train_unique(model,data,l_rate,epochs=100):

	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	epoch_losses = []
	criterion = nn.CrossEntropyLoss() 
	for epoch in range(epochs):
		epoch_l,el_z,el_recon,el_batch = (0,)*4
		for x_c1,y,x_zc,batch in data:
			opt.zero_grad()
			z_u,px_s,px_r,px_d,batch_pred = model(x_c1,x_zc)
			train_loss_z = minimal_overlap_loss(x_zc,z_u)
			train_loss_recon = get_zinb_reconstruction_loss(x_c1,px_s, px_r, px_d)
			train_loss_batch = criterion(batch_pred, batch)
			train_loss = train_loss_z + train_loss_recon + train_loss_batch
			train_loss.backward()

			opt.step()
			epoch_l += train_loss.item()
			el_z += train_loss_z.item()
			el_recon += train_loss_recon.item()
			el_batch += train_loss_batch.item()
		   
		epoch_losses.append([epoch_l/len(data),el_z/len(data),el_recon/len(data),el_batch/len(data)])  
		
		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch,epoch_l/len(data) ))

	return epoch_losses
 

def grasp_train_unique_gnn(model,data,l_rate,epochs,device):

	logger.info('Init training....nn_unq_graph')
	model.train()
	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	epoch_losses = []
	criterion = nn.CrossEntropyLoss() 
	
	x_graph = data.x_data  
	x_data_loader = RandomNodeLoader(x_graph, num_parts=25, shuffle=True)
	
	epoch_losses = []
	for epoch in range(epochs):
		epoch_l, el_z, el_recon, el_batch = 0, 0, 0, 0
		for batch in x_data_loader:
			opt.zero_grad()
   
			x_zc = data.x_zc.x[batch.y].to(device)
			x_c1 = batch.x.to(device)
			edge_index = batch.edge_index.to(device)
			batch_labels = data.batch_labels.x[batch.y].to(device)
  
			z_b, z_r, px_s, px_r, px_d, batch_pred = model(x_c1, x_zc, edge_index)
			train_loss_z = minimal_overlap_loss(z_b,z_r)
			train_loss_recon = get_zinb_reconstruction_loss(x_c1,px_s, px_r, px_d)
			train_loss_batch = criterion(batch_pred, batch_labels)
			train_loss = train_loss_z + train_loss_recon + train_loss_batch
			train_loss.backward()

			opt.step()
			epoch_l += train_loss.item()
			el_z += train_loss_z.item()
			el_recon += train_loss_recon.item()
			el_batch += train_loss_batch.item()
		   
		epoch_losses.append([epoch_l/len(data),el_z/len(data),el_recon/len(data),el_batch/len(data)])  
		
		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch,epoch_l/len(data) ))

	return epoch_losses
 