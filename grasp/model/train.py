import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import  minimal_overlap_loss, get_zinb_reconstruction_loss
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
		epoch_l, el_z, el_recon, el_batch = 0, 0, 0, 0
  
		for x_be_batch in x_be_loader:
	  
			x_c1 = x_be_batch.x.to(device)
			x_be_edge_index = x_be_batch.edge_index.to(device)
			x_indxs = x_be_batch.y

			x_zc = dataset.x_zc_data.x[x_indxs,:].to(device)
			batch_labels = dataset.x_batch_data.x[x_indxs].to(device)
   
			mask_0 = torch.isin(dataset.x_ge_data.x[0],x_indxs)
			mask_1 = torch.isin(dataset.x_ge_data.x[1],x_indxs)
			mask = mask_0 & mask_1
   
			x_ge_edge_index = dataset.x_ge_data.x[:,mask]
			index_map = {value.item(): idx for idx, value in enumerate(x_indxs)}
			x_ge_edge_index_be_space = torch.tensor([[index_map[node.item()] for node in edge] for edge in x_ge_edge_index])
			x_ge_edge_index_be_space = x_ge_edge_index_be_space.to(device)
   
			opt.zero_grad()
   
			z_be, z_ge, px_s, px_r, px_d, batch_pred = model(x_c1, x_zc, x_be_edge_index, x_ge_edge_index_be_space)
   
			train_loss_z = minimal_overlap_loss(z_be,z_ge)
			train_loss_recon = get_zinb_reconstruction_loss(x_c1,px_s, px_r, px_d)
			train_loss_batch = criterion(batch_pred, batch_labels)
			train_loss = train_loss_z + train_loss_recon + train_loss_batch
			train_loss.backward()

			opt.step()
			epoch_l += train_loss.item()
			el_z += train_loss_z.item()
			el_recon += train_loss_recon.item()
			el_batch += train_loss_batch.item()
		   
		epoch_losses.append([epoch_l/len(dataset),el_z/len(dataset),el_recon/len(dataset),el_batch/len(dataset)])  
		
		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch,epoch_l/len(dataset) ))

	return epoch_losses
 
 
 
	#  def train_epoch(self, args, model, device, dataset, optimizer, m, epoch):
	# 	""" Train for 1 epoch."""
	# 	model.train()
	# 	bce = nn.BCELoss()
	# 	ce = MarginLoss(m=-m)
	# 	sum_loss = 0

	# 	labeled_graph, unlabeled_graph = dataset.labeled_data, dataset.unlabeled_data
	# 	labeled_data = ClusterData(labeled_graph, num_parts=100, recursive=False)
	# 	labeled_loader = ClusterLoader(labeled_data, batch_size=1, shuffle=True,
	# 								num_workers=1)
	# 	unlabeled_data = ClusterData(unlabeled_graph, num_parts=100, recursive=False)
	# 	unlabeled_loader = ClusterLoader(unlabeled_data, batch_size=1, shuffle=True,
	# 								num_workers=1)
	# 	unlabel_loader_iter = cycle(unlabeled_loader)

	# 	for batch_idx, labeled_x in enumerate(labeled_loader):
	# 		unlabeled_x = next(unlabel_loader_iter)
	# 		unlabeled_ce_idx = torch.where(unlabeled_x.novel_label_seeds>0)[0]
	# 		labeled_x, unlabeled_x = labeled_x.to(device), unlabeled_x.to(device)
	# 		optimizer.zero_grad()
	# 		labeled_output, labeled_feat, _ = model(labeled_x)
	# 		unlabeled_output, unlabeled_feat, _ = model(unlabeled_x)
	# 		labeled_len = len(labeled_output)
	# 		batch_size = len(labeled_output) + len(unlabeled_output)
	# 		output = torch.cat([labeled_output, unlabeled_output], dim=0)
	# 		feat = torch.cat([labeled_feat, unlabeled_feat], dim=0)
			
	# 		prob = F.softmax(output, dim=1)
	# 		# Similarity labels
	# 		feat_detach = feat.detach()
	# 		feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
	# 		cosine_dist = torch.mm(feat_norm, feat_norm.t())

	# 		pos_pairs = []
	# 		target = labeled_x.y
	# 		target_np = target.cpu().numpy()
			
	# 		for i in range(labeled_len):
	# 			target_i = target_np[i]
	# 			idxs = np.where(target_np == target_i)[0]
	# 			if len(idxs) == 1:
	# 				pos_pairs.append(idxs[0])
	# 			else:
	# 				selec_idx = np.random.choice(idxs, 1)
	# 				while selec_idx == i:
	# 					selec_idx = np.random.choice(idxs, 1)
	# 				pos_pairs.append(int(selec_idx))
			
	# 		unlabel_cosine_dist = cosine_dist[labeled_len:, :]
	# 		vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
	# 		pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
	# 		pos_pairs.extend(pos_idx)
			
	# 		pos_prob = prob[pos_pairs, :]
	# 		pos_sim = torch.bmm(prob.view(batch_size, 1, -1), pos_prob.view(batch_size, -1, 1)).squeeze()
	# 		ones = torch.ones_like(pos_sim)
	# 		bce_loss = bce(pos_sim, ones)
	# 		ce_idx = torch.cat((torch.arange(labeled_len), labeled_len+unlabeled_ce_idx))
	# 		target = torch.cat((target, unlabeled_x.novel_label_seeds))
	# 		ce_loss = ce(output[ce_idx], target[ce_idx])
	# 		entropy_loss = entropy(torch.mean(prob, 0))
			
	# 		loss = 1 * bce_loss + 1 * ce_loss - 0.3 * entropy_loss

	# 		optimizer.zero_grad()
	# 		sum_loss += loss.item()
	# 		loss.backward()
	# 		optimizer.step()

	# 	print('Loss: {:.6f}'.format(sum_loss / (batch_idx + 1)))

