
import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import picasa
import torch
import logging


import glob
import os

sample = 'sim4'
wdir = 'figures/sim4/'

directory = wdir+'/data'
pattern = 'sim4_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('sim4_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >25:
		break


for batch_name, adata in batch_map.items():
    n_obs = adata.shape[0]

# Concatenate all AnnData objects
combined_adata = an.concat([adata for adata in batch_map.values()], merge='unique', uns_merge='unique')


adata = combined_adata

import scanpy as sc
import matplotlib.pyplot as plt
sc.pp.filter_cells(adata, min_genes=25)
sc.pp.filter_genes(adata, min_cells=2)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
adata = adata[:, adata.var.highly_variable]
sc.tl.pca(adata,random_state=42)


sc.pp.neighbors(adata)
sc.tl.leiden(adata,resolution=1.0)
sc.tl.umap(adata,min_dist=0.1)


dfl = pd.read_csv(wdir+'data/sim4_label.csv.gz')
dfl.rename(columns={'index':'cell'},inplace=True)
adata.obs = pd.merge(adata.obs,dfl,left_index=True,right_on='cell',how='left')

sc.pl.umap(adata, color=["leiden","Batch","Group"])
plt.savefig(wdir+'results/scanpy_umap.png')


import bbknn
adata_bbknn = adata.copy()
bbknn.bbknn(adata_bbknn, batch_key='Batch')
sc.tl.leiden(adata_bbknn,resolution=1.0)
sc.tl.umap(adata_bbknn,min_dist=0.1)
sc.pl.umap(adata_bbknn, color=["leiden","Batch","Group"])
plt.savefig(wdir+'results/scanpy_umap_bbknn.png')


adata_combat = adata.copy()
sc.pp.combat(adata_combat, key='Batch')
sc.pp.pca(adata_combat, n_comps=30, use_highly_variable=True, svd_solver='arpack')
sc.pp.neighbors(adata_combat, n_pcs =30)
sc.tl.umap(adata_combat)
sc.pl.umap(adata_combat, color=["leiden","Batch","Group"])
plt.savefig(wdir+'results/scanpy_umap_combat.png')


adata_mnn = adata.copy()
sc.external.pp.mnn_correct(adata_mnn, batch_key='Batch')
sc.pp.neighbors(adata_mnn, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata_mnn)
sc.tl.leiden(adata_mnn, resolution=1.0)
sc.pl.umap(adata_mnn, color=["leiden", "Batch","Group"])
plt.savefig(wdir+'results/scanpy_umap_mnn.png')