from .util.typehint import Adata
from . import dutil 
from . import model 
from . import neighbour
import torch
import logging
import gc
import os
import pandas as pd
import numpy as np
import itertools

from torch_geometric.loader import RandomNodeLoader


class grasp(object):
	def __init__(self, 
		data: dutil.data.Dataset, 
		pair_mode:str,
		wdir: str
		):
	 
		self.data = data
		self.wdir = wdir
		logging.basicConfig(filename=self.wdir+'results/4_attncl_train.log',
		format='%(asctime)s %(levelname)-8s %(message)s',
		level=logging.INFO,
		datefmt='%Y-%m-%d %H:%M:%S')
  
		self.adata_keys = list(self.data.adata_list.keys())
		 
		if pair_mode == 'unq':
			indices = range(len(self.data.adata_list))
			adata_pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices)-1, 1)]
			adata_pairs.append((adata_pairs[len(adata_pairs)-1][1],adata_pairs[0][0]))
			self.adata_pairs = adata_pairs
		else:
			indices = list(range(len(self.data.adata_list)))
			self.adata_pairs = list(itertools.combinations(indices, 2))
	
	def get_edges(self,
		):
		logging.info("generating global graph...")

		for adata_pair in self.adata_pairs:break
  
		p1 = self.adata_keys[adata_pair[0]]
		p2 = self.adata_keys[adata_pair[1]]

		adata_p1 = self.data.adata_list[p1]
		adata_p2 = self.data.adata_list[p2]
	
		nbr_df = self.grasp_common.uns['nbr_map']
		nbr_df = nbr_df[nbr_df['batch_pair']==p1+'_'+p2]
		nbr_map = {x:(y,z) for x,y,z in zip(nbr_df['key'],nbr_df['neighbor'],nbr_df['score'])}

		eval_batch_size = 100
		eval_total_size = 1000
		device = 'cuda'

		attn = self.get_gene_attention_estimate(adata_p1,adata_p2,nbr_map,eval_batch_size,eval_total_size,device)

		return attn.cpu().detach().numpy()

	def train_unique_gnn(self,
		edge_list:list,
		input_dim:int,
		enc_layers:list,
		common_latent_dim:int,
		unique_latent_dim:int,
		dec_layers:list,
		l_rate:float,
		epochs:int,
		batch_size:int,
		device:str
		):

		num_batches = len(self.adata_keys)
		grasp_unq_model = model.nn_gnn.graspUNET(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(device)
	
		logging.info(grasp_unq_model)
  
		x_c1_batches = []
		y_batches = []
		x_zc_batches = []
		b_ids_batches = []
  
		logging.info("Creating dataloader for training unique space.")
  
		for batch in self.adata_keys:
			adata_x = self.data.adata_list[batch]
			adata_zcommon = self.grasp_common[self.grasp_common.obs['batch']==batch]

			data = dutil.nn_load_data_with_latent(adata_x,adata_zcommon,batch,'cpu',batch_size)
	
			for x_c1,y,x_zc in data:		
				x_c1_batches.append(x_c1)
				y_batches.append(np.array(y))
				x_zc_batches.append(x_zc)
				b_ids_batches.append(torch.tensor([ self.batch_mapping[y_id] for y_id in y]))
	
		all_x_c1 = torch.cat(x_c1_batches, dim=0).float()  
		all_x_idxs = torch.tensor([ x for x in range(all_x_c1.shape[0])])        
		all_x_zc = torch.cat(x_zc_batches, dim=0)  
		all_b_ids = torch.cat(b_ids_batches,dim=0)  

		train_data =dutil.GraphDataset(all_x_c1,all_x_idxs,all_x_zc,edge_list,all_b_ids)

		logging.info('Training...unique space model.')
		loss = model.nn_gnn.train(grasp_unq_model,train_data,l_rate,epochs,device)

		torch.save(grasp_unq_model.state_dict(),self.wdir+'results/nn_unq.model')
		pd.DataFrame(loss,columns=['ep_l','el_z','el_recon','el_batch']).to_csv(self.wdir+'results/4_unq_train_loss.txt.gz',index=False,compression='gzip',header=True)
		logging.info('Completed training...model saved in results/nn_unq.model')

	def eval_unique_gnn(self,
        edge_list:list,	 
		input_dim:int,
		enc_layers:list,
		common_latent_dim:int,
		unique_latent_dim:int,
		dec_layers:list,
		eval_batch_size:int, 
		eval_total_size:int,
		device:str
		):

		num_batches = len(self.adata_keys)
		grasp_unq_model = model.nn_gnn.graspUNET(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(device)
	
		grasp_unq_model.load_state_dict(torch.load(self.wdir+'results/nn_unq.model', map_location=torch.device(device)))

		grasp_unq_model.eval()
  
		x_c1_batches = []
		y_batches = []
		x_zc_batches = []
		b_ids_batches = []
  
		logging.info("Creating dataloader for training unique space.")
  
		for batch in self.adata_keys:
			adata_x = self.data.adata_list[batch]
			adata_zcommon = self.grasp_common[self.grasp_common.obs['batch']==batch]

			data = dutil.nn_load_data_with_latent(adata_x,adata_zcommon,batch,'cpu',eval_batch_size)
	
			for x_c1,y,x_zc in data:		
				x_c1_batches.append(x_c1)
				y_batches.append(np.array(y))
				x_zc_batches.append(x_zc)
				b_ids_batches.append(torch.tensor([ self.batch_mapping[y_id] for y_id in y]))
	
		all_x_c1 = torch.cat(x_c1_batches, dim=0).float()  
		all_x_idxs = torch.tensor([ x for x in range(all_x_c1.shape[0])])        
		all_x_zc = torch.cat(x_zc_batches, dim=0)  
		all_b_ids = torch.cat(b_ids_batches,dim=0)

		y_batches_n = []
		for arr in y_batches:
			for yl in arr:
				y_batches_n.append(yl)
		
		y_batches = np.array(y_batches_n)

		data =dutil.GraphDataset(all_x_c1,all_x_idxs,all_x_zc,edge_list,all_b_ids)

		x_graph = data.x_data  
		x_data_loader = RandomNodeLoader(x_graph, num_parts=25, shuffle=True)

		df_u_latent = pd.DataFrame()
  
		for batch in x_data_loader:
			   
			x_zc = data.x_zc.x[batch.y].to(device)
			x_c1 = batch.x.to(device)
			y = batch.y.to(device)
			edge_index = batch.edge_index.to(device)

			z,ylabel = model.nn_gnn.predict_batch(grasp_unq_model,x_c1,y,x_zc,edge_index)
			z_u = z[0]

			ylabel = ylabel.cpu().detach().numpy()

			ylabel_name = [y_batches[x] for x in ylabel]
   
			df_u_latent = pd.concat([df_u_latent,pd.DataFrame(z_u.cpu().detach().numpy(),index=ylabel_name)],axis=0)

			if df_u_latent.shape[0]>eval_total_size:
				break
		return df_u_latent
   
	def save_common(self):
		import anndata as an
  
		batches = self.latent.keys()
		df = pd.DataFrame()

		for b in batches:
			c_df = self.latent[b]
			c_df.index = [x+'@'+b for x in c_df.index.values]

			df = pd.concat([df,c_df])

		df.columns = ['common_'+str(x) for x in df.columns]
		adata = an.AnnData(X=df)
		batch_loc = len(df.index.values[0].split('@'))-1
		adata.obs['batch'] = [x.split('@')[batch_loc] for x in df.index.values]

		adata.uns['adata_keys'] = self.adata_keys
		adata.uns['adata_pairs'] = self.adata_pairs
		adata.uns['nn_params'] = self.nn_params
		
		nbr_map_df = pd.DataFrame([
			{'batch_pair': l1_item, 'key': k, 'neighbor': v[0], 'score': v[1]}
			for l1_item, inner_map in self.nbr_map.items()
			for k, v in inner_map.items()
		])
		adata.uns['nbr_map'] = nbr_map_df
		adata.write(self.wdir+'results/grasp.h5ad')
  
	def plot_loss(self,
		tag:str
		):
		from grasp.util.plots import plot_loss
		if tag=='common':
			plot_loss(self.wdir+'results/4_attncl_train_loss.txt.gz',self.wdir+'results/4_attncl_train_loss.png')
		elif tag=='unq':
			plot_loss(self.wdir+'results/4_unq_train_loss.txt.gz',self.wdir+'results/4_unq_attncl_train_loss.png')


def create_grasp_object(
	adata_list:Adata, 
	pair_mode:str,
	wdir:str
	):
	return grasp(dutil.data.Dataset(adata_list),pair_mode,wdir)
