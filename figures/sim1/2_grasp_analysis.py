import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/grasp')


import grasp
import anndata as an
import pandas as pd



sample = 'sim1'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/grasp/figures/'+sample

############ read original data as adata list

ddir = wdir+'/data/'
batch1 = an.read_h5ad(ddir+sample+'_Batch1.h5ad')
batch2 = an.read_h5ad(ddir+sample+'_Batch2.h5ad')
batch3 = an.read_h5ad(ddir+sample+'_Batch3.h5ad')

grasp_data = {'Batch1':batch1,
	 'Batch2':batch2,
	 'Batch3':batch3
	 }


############ read model results as adata 
grasp_adata = an.read_h5ad(wdir+'/results/grasp.h5ad')



import scanpy as sc
import matplotlib.pylab as plt


sc.pp.neighbors(grasp_adata,use_rep='base')
sc.tl.umap(grasp_adata)
sc.tl.leiden(grasp_adata)
sc.pl.umap(grasp_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/grasp_base_umap.png')

sc.pp.neighbors(grasp_adata,use_rep='batch')
sc.tl.umap(grasp_adata)
sc.tl.leiden(grasp_adata)
sc.pl.umap(grasp_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/grasp_batch_umap.png')

sc.pp.neighbors(grasp_adata,use_rep='group')
sc.tl.umap(grasp_adata)
sc.tl.leiden(grasp_adata)
sc.pl.umap(grasp_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/grasp_group_umap.png')

