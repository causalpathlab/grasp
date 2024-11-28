
import anndata as an
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt

wdir = 'znode/pbmc/'

df = pd.read_csv(wdir+'data/pbmc_count.csv.gz',header=0)
df = df.T
df.columns = df.iloc[0,:]
df = df.iloc[1:,:]
df = df.astype(int)



smat = csr_matrix(df.to_numpy())
adata = an.AnnData(X=smat)
adata.var_names = df.columns.values
adata.obs_names = df.index.values
adata.obs['batch'] = [ x.split('_')[0] for x in df.index.values]


dfl = pd.read_csv(wdir+'data/pbmc_label.csv.gz',header=0)
dfl.columns = ['cell','celltype','batch']

adata.obs['celltype'] = pd.merge(df,dfl,left_index=True,right_on=['cell'],how='left')['celltype'].values

adata.obs.celltype.value_counts()
adata.obs.batch.value_counts()


batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df = adata_c.to_df()

    smat = csr_matrix(df.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df.columns.values
    adata_b.obs_names = df.index.values
    adata_b.obs['batch'] = adata_c.obs['batch'].values
    adata_b.obs['celltype'] = adata_c.obs['celltype'].values

    adata_b.write(wdir+'pbmc_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs.reset_index()
dfl.columns = ['cell','batch','celltype']
dfl.to_csv(wdir+'pbmc_label.csv.gz',compression='gzip')


