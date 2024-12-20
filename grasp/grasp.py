from . import dutil 
from . import model 
import torch
import logging
import gc
import pandas as pd
import anndata as an
import numpy as np
import itertools



class grasp(object):
	"""
	Initialize grasp model

	Parameters
	----------
	data: dict of batch_name:anndata, each anndata is separate batch
	wdir: path to save model outputs            
	sample: name of data sample            
				
	Returns
	-------
	None

	"""
	def __init__(self, 
		data: dutil.data.Dataset, 
		sample: str,
		wdir: str
		):
	 
	 
		self.data = data
		self.sample = sample
		self.wdir = wdir+sample
		
		# dutil.data.create_model_directories(self.wdir,['results'])
		
		print(self.wdir)
		logging.basicConfig(filename=self.wdir+'/results/grasp_model.log',
		format='%(asctime)s %(levelname)-8s %(message)s',
		level=logging.INFO,
		datefmt='%Y-%m-%d %H:%M:%S')
  
		self.adata_keys = list(self.data.adata_list.keys())
		 
		   
	def set_nn_params(self,
		params: dict
		):
		self.nn_params = params
  
	def set_batch_mapping(self):
		
		batch_ids_map = {label: idx for idx, label in enumerate(self.adata_keys)}
		batch_ids = []
		cell_ids = []
		for ak in self.adata_keys:
			cadata = self.data.adata_list[ak]
			batch_id = [batch_ids_map[x] for x in cadata.obs['batch']]
			cell_id = [x+'@'+ak for x in cadata.obs.index.values]
			batch_ids.append(batch_id)
			cell_ids.append(cell_id)
		batch_ids = np.concatenate([np.array(sublist) for sublist in batch_ids])
		cell_ids = np.concatenate([np.array(sublist) for sublist in cell_ids])
		
		batch_mapping = { id:blabel for id, blabel in zip(cell_ids,batch_ids)}

		self.batch_mapping = batch_mapping

	def set_metadata(self):
		
		df_meta = pd.DataFrame()
		for ad_name in self.data.adata_list:
			ad = self.data.adata_list[ad_name]
			df_meta = pd.concat([df_meta,ad.obs])
		
		df_meta.index = [x+'@'+y for x,y in zip(df_meta.index.values,df_meta['batch'])]
		sel_col = [ x for x in df_meta.columns if x not in ['batch','batch_id']]
		
		self.result.obs = pd.merge(self.result.obs,df_meta[sel_col],left_index=True,right_index=True,how='left')
		
	def train_base(self,
		input_dim:int,
		enc_layers:list,
		latent_dim:int,
		dec_layers:list,
		l_rate:float,
		epochs:int,
		batch_size:int,
		device:str
		):

		num_batches = len(self.adata_keys)
		grasp_base_model = model.GRASPBaseNet(input_dim,latent_dim,enc_layers,dec_layers,num_batches).to(device)
	
		logging.info(grasp_base_model)
  
		x_c1_batches = []
		y_batches = []
		b_ids_batches = []
  
		logging.info("Creating dataloader for training grasp base model.")
  
		for batch in self.adata_keys:
			adata_x = self.data.adata_list[batch]

			data = dutil.nn_load_data_base(adata_x,batch,'cpu',batch_size)
	
			for x_c1,y in data:		
				x_c1_batches.append(x_c1)
				y_batches.append(np.array(y))
				b_ids_batches.append(torch.tensor([ self.batch_mapping[y_id] for y_id in y]))
	
		all_x_c1 = torch.cat(x_c1_batches, dim=0)  
		all_y = np.concatenate(y_batches)        
		all_b_ids = torch.cat(b_ids_batches,dim=0)  

		train_data =dutil.get_dataloader_mem_base(all_x_c1,all_y,all_b_ids,batch_size,device)

		logging.info('Training... GRASP unique model.')
		loss = model.grasp_train_base(grasp_base_model,train_data,l_rate,epochs)

		torch.save(grasp_base_model.state_dict(),self.wdir+'/results/grasp_base.model')
		pd.DataFrame(loss,columns=['ep_l','el_z','el_recon','el_batch']).to_csv(self.wdir+'/results/grasp_base_train_loss.txt.gz',index=False,compression='gzip',header=True)
		logging.info('Completed training...model saved in '+self.wdir+'/results/grasp_base.model')
  
	def eval_base(self,	 
		input_dim:int,
		enc_layers:list,
		latent_dim:int,
		dec_layers:list,
		eval_batch_size:int, 
		device:str
		):

		num_batches = len(self.adata_keys)
		grasp_base_model = model.GRASPBaseNet(input_dim,latent_dim,enc_layers,dec_layers,num_batches).to(device)
	
		grasp_base_model.load_state_dict(torch.load(self.wdir+'/results/grasp_base.model', map_location=torch.device(device)))

		grasp_base_model.eval()
  
		x_c1_batches = []
		y_batches = []
		b_ids_batches = []
  
		logging.info("Creating dataloader for evaluating grasp unique model.")
  
		for batch in self.adata_keys:
			adata_x = self.data.adata_list[batch]

			data = dutil.nn_load_data_base(adata_x,batch,'cpu',eval_batch_size)
	
			for x_c1,y in data:		
				x_c1_batches.append(x_c1)
				y_batches.append(np.array(y))
				b_ids_batches.append(torch.tensor([ self.batch_mapping[y_id] for y_id in y]))
	
		all_x_c1 = torch.cat(x_c1_batches, dim=0)  
		all_y = np.concatenate(y_batches)        
		all_b_ids = torch.cat(b_ids_batches,dim=0)  

		train_data =dutil.get_dataloader_mem_base(all_x_c1,all_y,all_b_ids,eval_batch_size,device)
			
		df_latent = pd.DataFrame()
  
		for x_c1,y,b_id in train_data:
			z,ylabel = model.predict_batch_base(grasp_base_model,x_c1,y)
			z_u = z[0]
			df_latent = pd.concat([df_latent,pd.DataFrame(z_u.cpu().detach().numpy(),index=ylabel)],axis=0)

		df_latent.columns = ['base_'+str(x) for x in df_latent.columns]
		adata = an.AnnData(obs=pd.DataFrame(index=df_latent.index))
		adata.obsm['base'] = df_latent
		
		batch_loc = len(df_latent.index.values[0].split('@'))-1
		adata.obs['batch'] = [x.split('@')[batch_loc] for x in df_latent.index.values]

		adata.uns['adata_keys'] = self.adata_keys
		self.result = adata
		self.set_metadata()
  

	def train_unique(self,
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
		grasp_unq_model = model.GRASPUniqueNet(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(device)
	
		logging.info(grasp_unq_model)
  
		x_c1_batches = []
		y_batches = []
		x_zc_batches = []
		b_ids_batches = []
  
		logging.info("Creating dataloader for training grasp unique model.")
  
		for batch in self.adata_keys:
			adata_x = self.data.adata_list[batch]
			df_zcommon = self.result.obsm['base'][self.result.obs['batch']==batch]

			data = dutil.nn_load_data_with_latent(adata_x,df_zcommon,batch,'cpu',batch_size)
	
			for x_c1,y,x_zc in data:		
				x_c1_batches.append(x_c1)
				y_batches.append(np.array(y))
				x_zc_batches.append(x_zc)
				b_ids_batches.append(torch.tensor([ self.batch_mapping[y_id] for y_id in y]))
	
		all_x_c1 = torch.cat(x_c1_batches, dim=0)  
		all_y = np.concatenate(y_batches)        
		all_x_zc = torch.cat(x_zc_batches, dim=0)  
		all_b_ids = torch.cat(b_ids_batches,dim=0)  

		train_data =dutil.get_dataloader_mem(all_x_c1,all_y,all_x_zc,all_b_ids,batch_size,device)

		logging.info('Training... GRASP unique model.')
		loss = model.grasp_train_unique(grasp_unq_model,train_data,l_rate,epochs)

		torch.save(grasp_unq_model.state_dict(),self.wdir+'/results/grasp_unique.model')
		pd.DataFrame(loss,columns=['ep_l','el_z','el_recon','el_batch']).to_csv(self.wdir+'/results/grasp_unique_train_loss.txt.gz',index=False,compression='gzip',header=True)
		logging.info('Completed training...model saved in '+self.wdir+'/results/grasp_unique.model')
  
	def eval_unique(self,	 
		input_dim:int,
		enc_layers:list,
		common_latent_dim:int,
		unique_latent_dim:int,
		dec_layers:list,
		eval_batch_size:int, 
		device:str
		):

		num_batches = len(self.adata_keys)
		grasp_unq_model = model.GRASPUniqueNet(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(device)
	
		grasp_unq_model.load_state_dict(torch.load(self.wdir+'/results/grasp_unique.model', map_location=torch.device(device)))

		grasp_unq_model.eval()
  
		x_c1_batches = []
		y_batches = []
		x_zc_batches = []
		b_ids_batches = []
  
		logging.info("Creating dataloader for evaluating grasp unique model.")
  
		for batch in self.adata_keys:
			adata_x = self.data.adata_list[batch]
			df_zcommon = self.result.obsm['base'][self.result.obs['batch']==batch]

			data = dutil.nn_load_data_with_latent(adata_x,df_zcommon,batch,'cpu',eval_batch_size)
	
			for x_c1,y,x_zc in data:		
				x_c1_batches.append(x_c1)
				y_batches.append(np.array(y))
				x_zc_batches.append(x_zc)
				b_ids_batches.append(torch.tensor([ self.batch_mapping[y_id] for y_id in y]))
	
		all_x_c1 = torch.cat(x_c1_batches, dim=0)  
		all_y = np.concatenate(y_batches)        
		all_x_zc = torch.cat(x_zc_batches, dim=0)  
		all_b_ids = torch.cat(b_ids_batches,dim=0)  

		train_data =dutil.get_dataloader_mem(all_x_c1,all_y,all_x_zc,all_b_ids,eval_batch_size,device)
			
		df_u_latent = pd.DataFrame()
  
		for x_c1,y,x_zc,b_id in train_data:
			z,ylabel = model.predict_batch_unique(grasp_unq_model,x_c1,y,x_zc)
			z_u = z[0]
			df_u_latent = pd.concat([df_u_latent,pd.DataFrame(z_u.cpu().detach().numpy(),index=ylabel)],axis=0)

		df_u_latent = df_u_latent.loc[self.result.obsm['base'].index.values,:]
		df_u_latent.columns = ['unique_'+str(x) for x in df_u_latent.columns]
		self.result.obsm['unique'] = df_u_latent

	def train_unique_gnn(self,
		edge_list:list,
		edge_list_sec:list,
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
		grasp_unq_model = model.GRASPUniqueGNET(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(device)
	
		logging.info(grasp_unq_model)
  
		x_c1_batches = []
		y_batches = []
		x_zc_batches = []
		b_ids_batches = []
  
		logging.info("Creating dataloader for training unique space.")
  
		for batch in self.adata_keys:
			adata_x = self.data.adata_list[batch]
			df_zcommon = self.result.obsm['base'][self.result.obs['batch']==batch]

			data = dutil.nn_load_data_with_latent(adata_x,df_zcommon,batch,'cpu',batch_size)
	
			for x_c1,y,x_zc in data:		
				x_c1_batches.append(x_c1)
				y_batches.append(np.array(y))
				x_zc_batches.append(x_zc)
				b_ids_batches.append(torch.tensor([ self.batch_mapping[y_id] for y_id in y]))
	
		all_x_c1 = torch.cat(x_c1_batches, dim=0).float()  
		all_x_idxs = torch.tensor([ x for x in range(all_x_c1.shape[0])])        
		all_x_zc = torch.cat(x_zc_batches, dim=0)  
		all_b_ids = torch.cat(b_ids_batches,dim=0)  

		train_data =dutil.GraphDataset(all_x_c1,all_x_idxs,all_x_zc,edge_list,edge_list_sec,all_b_ids)

		logging.info('Training...unique space model.')
		loss = model.grasp_train_unique_gnn(grasp_unq_model,train_data,l_rate,epochs,device)

		torch.save(grasp_unq_model.state_dict(),self.wdir+'/results/grasp_unique.model')
		pd.DataFrame(loss,columns=['ep_l','el_z','el_recon','el_batch']).to_csv(self.wdir+'/results/grasp_unique_train_loss.txt.gz',index=False,compression='gzip',header=True)
		logging.info('Completed training...model saved in /results/grasp_unique.model')

	def eval_unique_gnn(self,
		edge_list:list,	 
		edge_list_sec:list,	 
		input_dim:int,
		enc_layers:list,
		common_latent_dim:int,
		unique_latent_dim:int,
		dec_layers:list,
		eval_batch_size:int, 
		device:str
		):
	 
		from torch_geometric.loader import RandomNodeLoader


		num_batches = len(self.adata_keys)
		grasp_unq_model = model.GRASPUniqueGNET(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(device)
	
		grasp_unq_model.load_state_dict(torch.load(self.wdir+'/results/grasp_unique.model', map_location=torch.device(device)))

		grasp_unq_model.eval()
  
		x_c1_batches = []
		y_batches = []
		x_zc_batches = []
		b_ids_batches = []
  
		logging.info("Creating dataloader for training unique space.")
  
		for batch in self.adata_keys:
			adata_x = self.data.adata_list[batch]
			df_zcommon = self.result.obsm['base'][self.result.obs['batch']==batch]

			data = dutil.nn_load_data_with_latent(adata_x,df_zcommon,batch,'cpu',eval_batch_size)
	
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

		df_be_latent = pd.DataFrame()
		df_ge_latent = pd.DataFrame()
  
		for batch in x_data_loader:
			   
			x_zc = data.x_zc.x[batch.y].to(device)
			x_c1 = batch.x.to(device)
			y = batch.y.to(device)
			edge_index = batch.edge_index.to(device)

			z,ylabel = model.predict_batch_unique_gnn(grasp_unq_model,x_c1,y,x_zc,edge_index)
			z_be = z[0]
			z_ge = z[1]

			ylabel = ylabel.cpu().detach().numpy()

			ylabel_name = [y_batches[x] for x in ylabel]
   
			df_be_latent = pd.concat([df_be_latent,pd.DataFrame(z_be.cpu().detach().numpy(),index=ylabel_name)],axis=0)
			df_ge_latent = pd.concat([df_ge_latent,pd.DataFrame(z_ge.cpu().detach().numpy(),index=ylabel_name)],axis=0)

		df_be_latent = df_be_latent.loc[self.result.obsm['base'].index.values,:]
		df_be_latent.columns = ['batch_'+str(x) for x in df_be_latent.columns]
		self.result.obsm['batch'] = df_be_latent
  
		df_ge_latent = df_ge_latent.loc[self.result.obsm['base'].index.values,:]
		df_ge_latent.columns = ['group_'+str(x) for x in df_ge_latent.columns]
		self.result.obsm['group'] = df_ge_latent
  
   
	def plot_loss(self,
		tag:str
		):
		from grasp.util.plots import plot_loss
		if tag=='unq':
			plot_loss(self.wdir+'/results/grasp_unique_train_loss.txt.gz',self.wdir+'/results/grasp_unique_train_loss.png')
		elif tag=='base':
			plot_loss(self.wdir+'/results/grasp_base_train_loss.txt.gz',self.wdir+'/results/grasp_base_train_loss.png')
	
	def save_model(self):
		self.result.write(self.wdir+'/results/grasp.h5ad',compression='gzip')
		
def create_grasp_object(
	adata_list:an.AnnData,
	sample:str, 
	wdir:str
	):
	return grasp(dutil.data.Dataset(adata_list),sample,wdir)
