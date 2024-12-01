from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torch
import numpy as np
import logging
logger = logging.getLogger(__name__)


class SparseData():
	def __init__(self,x_indptr,x_indices,x_vals,x_shape,x_label):
		self.x_indptr = x_indptr
		self.x_indices = x_indices
		self.x_vals = x_vals
		self.x_shape = x_shape
		self.x_label = x_label

class SparseDataset(Dataset):
	def __init__(self, sparse_data,device):
		self.x_indptr = sparse_data.x_indptr
		self.x_indices = sparse_data.x_indices
		self.x_vals = sparse_data.x_vals
		self.x_shape = sparse_data.x_shape
		self.x_label = sparse_data.x_label
		self.device = device

	def __len__(self):
		return self.x_shape[0]

	def __getitem__(self, idx):

		x_cell = torch.zeros((self.x_shape[1],), dtype=torch.int32, device=self.device)
		x_ind1,x_ind2 = self.x_indptr[idx],self.x_indptr[idx+1]
		x_cell[self.x_indices[x_ind1:x_ind2].long()] = self.x_vals[x_ind1:x_ind2]

		return x_cell, self.x_label[idx]

def nn_load_data_base(adata_x,batch_name,device,batch_size):

	device = torch.device(device)
   
	x_indptr = torch.tensor(adata_x.X.indptr.astype(np.int32), dtype=torch.int32, device=device)
	x_indices = torch.tensor(adata_x.X.indices.astype(np.int32), dtype=torch.int32, device=device)
	x_vals = torch.tensor(adata_x.X.data.astype(np.int32), dtype=torch.int32, device=device)
	x_shape = tuple(adata_x.X.shape)
	x_label = [x+'@'+batch_name for x in adata_x.obs.index.values]
 
	spdata = SparseData(x_indptr,x_indices,x_vals,x_shape,x_label)

	return DataLoader(SparseDataset(spdata,device), batch_size=batch_size, shuffle=True)
