import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/grasp')

import grasp
import anndata as an



sample = 'sim4'
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
    'sim4',
	wdir
 	)


for adn in grasp_object.data.adata_list:
    ad = grasp_object.data.adata_list[adn]
    ad.obs['batch'] = [ adn for x in ad.obs.index.values ]
    
grasp_object.set_batch_mapping()


input_dim = grasp_object.data.adata_list['Batch1'].X.shape[1]
enc_layers = [128,15]
dec_layers = [128,128]





latent_dim=15
grasp_object.train_base(input_dim, enc_layers,latent_dim,dec_layers,l_rate=0.001,epochs=250,batch_size=128,device='cuda')
grasp_object.plot_loss(tag='base')

eval_batch_size = 10
grasp_object.eval_base(input_dim, enc_layers,latent_dim,dec_layers,eval_batch_size,device='cuda')


unique_latent_dim = 15
common_latent_dim = grasp_object.result.obsm['base'].shape[1]

### get graph
base_emb = grasp_object.result.obsm['base']
import numpy as np
attn = np.corrcoef(base_emb)
attn.shape


batch_labels = grasp_object.result.obs.batch.values

batch_mask = (batch_labels[:, None] != batch_labels) 
batch_mask = batch_mask.astype(float)
masked_correlation_matrix = attn * batch_mask
 
N=masked_correlation_matrix.shape[0]
total_edges = (N*(N-1))/2
for t in [1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.5,0.75,0.9]:
    ce = (masked_correlation_matrix>t).sum()
    print(t,ce,(ce/total_edges)*100)

# distance_thres = 0.8
# dists_mask = masked_correlation_matrix > distance_thres
dists_mask = masked_correlation_matrix
np.fill_diagonal(dists_mask, 0)
edge_list = np.transpose(np.nonzero(dists_mask)).tolist()


grasp_object.train_unique_gnn(edge_list,input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,l_rate=0.001,epochs=250,batch_size=128,device='cuda')
grasp_object.plot_loss(tag='unq')


eval_batch_size = 10
grasp_object.eval_unique_gnn(edge_list,input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,eval_batch_size,device='cuda')

grasp_object.save_model()
