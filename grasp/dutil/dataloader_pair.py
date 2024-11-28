from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torch
import numpy as np
import logging
logger = logging.getLogger(__name__)


class SparseData():
	def __init__(self,sc_indptr,sc_indices,sc_vals,sc_shape,sc_label,sp_indptr,sp_indices,sp_vals,sp_shape,scsp_map):
		self.sc_indptr = sc_indptr
		self.sc_indices = sc_indices
		self.sc_vals = sc_vals
		self.sc_shape = sc_shape
		self.sc_label = sc_label
		self.sp_indptr = sp_indptr
		self.sp_indices = sp_indices
		self.sp_vals = sp_vals
		self.sp_shape = sp_shape
		self.scsp_map = scsp_map

class SparseDataset(Dataset):
	def __init__(self, sparse_data,device):
		self.sc_indptr = sparse_data.sc_indptr
		self.sc_indices = sparse_data.sc_indices
		self.sc_vals = sparse_data.sc_vals
		self.sc_shape = sparse_data.sc_shape
		self.sc_label = sparse_data.sc_label
		self.sp_indptr = sparse_data.sp_indptr
		self.sp_indices = sparse_data.sp_indices
		self.sp_vals = sparse_data.sp_vals
		self.sp_shape = sparse_data.sp_shape
		self.scsp_map = sparse_data.scsp_map
		self.device = device

	def __len__(self):
		return self.sc_shape[0]

	def __getitem__(self, idx):

		sc_cell = torch.zeros((self.sc_shape[1],), dtype=torch.int32, device=self.device)
		sc_ind1,sc_ind2 = self.sc_indptr[idx],self.sc_indptr[idx+1]
		sc_cell[self.sc_indices[sc_ind1:sc_ind2].long()] = self.sc_vals[sc_ind1:sc_ind2]

		sp_pos_idx_value = self.scsp_map[idx]
		sp_pos_idx = sp_pos_idx_value[0]
		sp_pos_value = sp_pos_idx_value[1]
  
		sp_pos_cell = torch.zeros((self.sp_shape[1],), dtype=torch.int32, device=self.device)
		sp_pos_ind1,sp_pos_ind2 = self.sp_indptr[sp_pos_idx],self.sp_indptr[sp_pos_idx+1]
		sp_pos_cell[self.sp_indices[sp_pos_ind1:sp_pos_ind2].long()] = self.sp_vals[sp_pos_ind1:sp_pos_ind2]

		return sc_cell, self.sc_label[idx], sp_pos_cell, sp_pos_value

def nn_load_data(adata_sc,adata_sp,scsp_map,device,bath_size):


	device = torch.device(device)
 
	sc_indptr = torch.tensor(adata_sc.X.indptr.astype(np.int32), dtype=torch.int32, device=device)
	sc_indices = torch.tensor(adata_sc.X.indices.astype(np.int32), dtype=torch.int32, device=device)
	sc_vals = torch.tensor(adata_sc.X.data.astype(np.int32), dtype=torch.int32, device=device)
	sc_shape = tuple(adata_sc.X.shape)
	sc_label = adata_sc.obs.index.values

	sp_indptr = torch.tensor(adata_sp.X.indptr.astype(np.int32), dtype=torch.int32, device=device)
	sp_indices = torch.tensor(adata_sp.X.indices.astype(np.int32), dtype=torch.int32, device=device)
	sp_vals = torch.tensor(adata_sp.X.data.astype(np.int32), dtype=torch.int32, device=device)
	sp_shape = tuple(adata_sp.X.shape)
 
	spdata = SparseData(sc_indptr,sc_indices,sc_vals,sc_shape,sc_label,sp_indptr,sp_indices,sp_vals,sp_shape,scsp_map)

	return DataLoader(SparseDataset(spdata,device), batch_size=bath_size, shuffle=True)
