import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/grasp')


import grasp
import anndata as an
import pandas as pd


pp = '/home/BCCRC.CA/ssubedi/projects/experiments/grasp'
sample = 'pbmc'


############ read original data as adata list

ddir = pp+'/figures/'+sample+'/data/'
batch1 = an.read_h5ad(ddir+sample+'_pbmc1.h5ad')
batch2 = an.read_h5ad(ddir+sample+'_pbmc2.h5ad')
grasp_data = {'pbmc1':batch1,'pbmc2':batch2}


############ read model results as adata 
wdir = pp+'/figures/'+sample
grasp_adata = an.read_h5ad(wdir+'/results/grasp.h5ad')


############ add metadata
dfl= pd.read_csv(ddir+sample+'_label.csv.gz')
dfl.columns = ['index','cell','batch','celltype']
dfl.cell = [x+'@'+y for x,y in zip(dfl['cell'],dfl['batch'])]
dfl = dfl[['index','cell','celltype']]
grasp_adata.obs = pd.merge(grasp_adata.obs,dfl,left_index=True,right_on='cell')



import scanpy as sc
import matplotlib.pylab as plt


sc.pp.neighbors(grasp_adata,use_rep='base')
sc.tl.umap(grasp_adata)
sc.tl.leiden(grasp_adata)
sc.pl.umap(grasp_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/grasp_base_umap.png')

sc.pp.neighbors(grasp_adata,use_rep='unique')
sc.tl.umap(grasp_adata)
sc.tl.leiden(grasp_adata)
sc.pl.umap(grasp_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/grasp_unique_umap.png')

