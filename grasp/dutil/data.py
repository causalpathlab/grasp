"""
Base classes for data definition 
"""

from typing import List, Mapping
from ..util.typehint import Adata

import numpy as np
from scipy.sparse import csr_matrix 
import h5py as hf

def configure_dataset(
    adata : Adata
) -> None:
    pass

    
class Dataset:
    def __init__(self, adata_list : Mapping[str, Adata]) -> None:
        self.adata_list = adata_list
            


def load_data(adata, start_index: int, end_index: int)-> np.array:
    
    selected_gene_indices = adata.uns['selected_genes']
    
    if isinstance(adata.X,np.ndarray):
        return adata.X[:,selected_gene_indices]
    
    elif isinstance(adata.X,csr_matrix):
        data = adata.X.data
        indices = adata.X.indices
        indptr = adata.X.indptr
        shape = adata.X.shape
        
        
        mtx = []
        for ci in range(start_index,end_index,1):
            mtx.append(np.asarray(
            csr_matrix((data[indptr[ci]:indptr[ci+1]], 
            indices[indptr[ci]:indptr[ci+1]], 
            np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), 
            shape=(1,shape[1])).todense()).flatten())
        
        return np.asarray(mtx)

def write_h5(fname,row_names,col_names,smat):

	f = hf.File(fname+'.h5','w')

	grp = f.create_group('matrix')

	grp.create_dataset('barcodes', data = row_names ,compression='gzip')

	grp.create_dataset('indptr',data=smat.indptr,compression='gzip')
	grp.create_dataset('indices',data=smat.indices,compression='gzip')
	grp.create_dataset('data',data=smat.data,compression='gzip')

	data_shape = np.array([len(row_names),len(col_names)])
	grp.create_dataset('shape',data=data_shape)
	
	f['matrix'].create_group('features')
	f['matrix']['features'].create_dataset('id',data=col_names,compression='gzip')

	f.close()
