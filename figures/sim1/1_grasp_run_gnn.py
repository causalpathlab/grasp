import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/grasp')

import grasp
import anndata as an



sample = 'sim1'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/grasp/figures/'
ddir = wdir+sample+'/data/'
batch1 = an.read_h5ad(ddir+sample+'_Batch1.h5ad')
batch2 = an.read_h5ad(ddir+sample+'_Batch2.h5ad')
batch3 = an.read_h5ad(ddir+sample+'_Batch3.h5ad')

grasp_object = grasp.create_grasp_object(
	{'Batch1':batch1,
	 'Batch2':batch2,
	 'Batch3':batch3
	 },
    'sim1',
	wdir
 	)

#### set group and batch labels
grasp_object.set_batch_group_mapping()

### base model learning
input_dim = grasp_object.data.adata_list['Batch1'].X.shape[1]
enc_layers = [128,15]
dec_layers = [128,128]
latent_dim=15
grasp_object.train_base(input_dim, enc_layers,latent_dim,dec_layers,l_rate=0.001,epochs=250,batch_size=128,device='cuda')
grasp_object.plot_loss(tag='base')
eval_batch_size = 1000
grasp_object.eval_base(input_dim, enc_layers,latent_dim,dec_layers,eval_batch_size,device='cuda')


### get graph
common_latent_dim = grasp_object.result.obsm['base'].shape[1]
base_emb = grasp_object.result.obsm['base']
import numpy as np
attn = np.corrcoef(base_emb)
attn.shape

### batch wise graph
batch_labels = grasp_object.result.obs.batch.values
batch_mask = (batch_labels[:, None] != batch_labels) 
batch_mask = batch_mask.astype(float)
np.fill_diagonal(batch_mask, 0)
masked_correlation_matrix = attn * batch_mask
 
N=masked_correlation_matrix.shape[0]
total_edges = (N*(N-1))/2
for t in [1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.5,0.75,0.9]:
    ce = (masked_correlation_matrix>t).sum()
    print(t,ce,(ce/total_edges)*100)

distance_thres = 0.1
dists_mask = masked_correlation_matrix > distance_thres
edge_list = np.transpose(np.nonzero(dists_mask))
print(edge_list)


### group wise graph
group_labels = grasp_object.result.obs.celltype.values
group_mask = (group_labels[:, None] != group_labels) 
group_mask = group_mask.astype(float)
group_mask.sum()
batch_mask.sum()
update_mask = np.logical_and(group_mask == 1, batch_mask == 1).astype(float)
update_mask.sum()
masked_correlation_matrix = attn * update_mask
N=masked_correlation_matrix.shape[0]
total_edges = (N*(N-1))/2
for t in [1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.5,0.75,0.9]:
    ce = (masked_correlation_matrix>t).sum()
    print(t,ce,(ce/total_edges)*100)

distance_thres = 0.1
dists_mask = masked_correlation_matrix > distance_thres
np.fill_diagonal(dists_mask, 0)
edge_list_sec = np.transpose(np.nonzero(dists_mask))


### grasp model training
unique_latent_dim = 15
grasp_object.train_unique_gnn(edge_list,edge_list_sec,input_dim, enc_layers,unique_latent_dim,l_rate=0.001,epochs=750,batch_size=128,device='cuda')
grasp_object.plot_loss(tag='unq')
eval_batch_size = 1000
grasp_object.eval_unique_gnn(edge_list,edge_list_sec,input_dim, enc_layers,unique_latent_dim,eval_batch_size,device='cuda')

grasp_object.save_model()
