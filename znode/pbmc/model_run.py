
import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/grasp')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import grasp
import torch
import logging

sample = 'pbmc'
wdir = 'znode/pbmc/'

batch1 = an.read_h5ad(wdir+'data/'+sample+'_pbmc1.h5ad')
batch2 = an.read_h5ad(wdir+'data/'+sample+'_pbmc2.h5ad')

grasp_object = grasp.pic.create_grasp_object(
	{'pbmc1':batch1,
	 'pbmc2':batch2
	 },'unq',
	wdir)

for batch_name in grasp_object.data.adata_list.keys():
    grasp_object.data.adata_list[batch_name].obs.index = [x+'@'+batch_name for x in grasp_object.data.adata_list[batch_name].obs.index.values]

grasp_common = an.read(wdir+'results/grasp.h5ad')

## assign batch
batch_ids = {label: idx for idx, label in enumerate(grasp_common.obs['batch'].unique())}
grasp_common.obs['batch_id'] = [batch_ids[x] for x in grasp_common.obs['batch']]
batch_mapping = { idx:label for idx, label in zip(grasp_common.obs.index.values,grasp_common.obs['batch_id'])}

grasp_object.set_batch_mapping(batch_mapping)

grasp_object.set_grasp_common(grasp_common)


params = {'device': 'cuda', 'batch_size': 128, 'input_dim': 1000, 'embedding_dim': 1000, 'attention_dim': 25, 'latent_dim': 15, 'encoder_layers': [100,15], 'projection_layers': [15, 15], 'learning_rate': 0.001, 'lambda_loss': [0.5, 0.1, 0.01, 1.0], 'temperature_cl': 1.0, 'pair_search_method': 'approx_50', 'pair_importance_weight': 0.0, 'corruption_tol':2, 'cl_loss_mode': 'weighted', 'loss_clusters': 5, 'loss_threshold': 0.1, 'loss_weight': 2.0, 'epochs': 1, 'titration': 50}

grasp_object.set_nn_params(params)

attn = grasp_object.get_edges()

N=1000
total_edges = (N*(N-1))/2
for t in [1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.5]:
    ce = (attn>t).sum()
    print(t,ce,(ce/total_edges)*100)

distance_thres = 1e-4
dists_mask = attn < distance_thres
np.fill_diagonal(dists_mask, 0)
edge_list = np.transpose(np.nonzero(dists_mask)).tolist()

edge_list = np.transpose(np.nonzero(attn)).tolist()




input_dim = grasp_object.data.adata_list['pbmc1'].X.shape[1]
enc_layers = [128,15]
unique_latent_dim = 15
common_latent_dim = grasp_common.X.shape[1]
dec_layers = [128,128]
epochs=2000
batch_size=128
l_rate=0.001
device='cuda'

grasp_object.train_unique_gnn(edge_list, input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,l_rate,epochs,batch_size,device)


grasp_object.plot_loss(tag='unq')

eval_batch_size = 10
eval_total_size = 10000

df_u = grasp_object.eval_unique_gnn(edge_list,input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,eval_batch_size, eval_total_size,device='cuda')
df_u.to_csv(wdir+'results/df_u.csv.gz',compression='gzip')




import umap 
from grasp.util.plots import plot_umap_df

sample = 'pbmc'
wdir = 'znode/pbmc/'
 
grasp_common = an.read(wdir+'results/grasp.h5ad')
df_c = grasp_common.to_df()
df_u = pd.read_csv(wdir+'results/df_u.csv.gz',index_col=0)

dfl = pd.read_csv(wdir+'data/pbmc_label.csv.gz')
dfl.columns = ['index','cell','batch','celltype']
dfl.cell = [x+'@'+y for x,y in zip(dfl['cell'],dfl['batch'])]

umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_c)
df_umap= pd.DataFrame()
df_umap['cell'] = df_c.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_c',pt_size=1.0,ftype='png')
df_umap['batch'] = pd.merge(df_umap,dfl,on='cell',how='left')['batch'].values
plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_lat_c_batch',pt_size=1.0,ftype='png')



umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_u)
df_umap= pd.DataFrame()
df_umap['cell'] = df_u.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_unq',pt_size=1.0,ftype='png')

df_umap['batch'] = pd.merge(df_umap,dfl,on='cell',how='left')['batch'].values
plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_lat_unq_batch',pt_size=1.0,ftype='png')


