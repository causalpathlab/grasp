import pandas as pd
import numpy as np
from plotnine import *
import seaborn as sns
import matplotlib.pylab as plt

from .analysis import get_topic_top_genes,row_col_order
from .palette import get_colors


def plot_umap_df(df_umap,col,fpath,pt_size=1.0,ftype='png'):
	
	nlabel = df_umap[col].nunique()
	custom_palette = get_colors(nlabel) 
 

	if ftype == 'pdf':
		fname = fpath+'_'+col+'_'+'umap.pdf'
	else:
		fname = fpath+'_'+col+'_'+'umap.png'
	
	legend_size = 7
		
	p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
		geom_point(size=pt_size) +
		scale_color_manual(values=custom_palette)  +
		guides(color=guide_legend(override_aes={'size': legend_size})))
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
	
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)

def plot_loss(loss_f,fpath,pt_size=4.0):
    
	plt.rcParams.update({'font.size': 20})
 
	data = pd.read_csv(loss_f)
	num_subplots = len(data.columns)
 
	if num_subplots>1:
		fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6*num_subplots), sharex=True)

		for i, column in enumerate(data.columns):
			data[[column]].plot(ax=axes[i], legend=None, linewidth=pt_size, marker='o') 
			axes[i].set_ylabel(column)
			axes[i].set_xlabel('epoch')
			axes[i].grid(False)

		plt.tight_layout()
	else:
		data[data.columns[0]].plot( legend=None, linewidth=pt_size, marker='o') 
	plt.savefig(fpath);plt.close()


def plot_gene_loading(df_beta,top_n,max_thresh,fname):
	df_beta = df_beta.loc[:, ~df_beta.columns.duplicated(keep='first')]
	df_top = get_topic_top_genes(df_beta.iloc[:,:],top_n)
	df_beta = df_beta.loc[:,df_top['Gene'].unique()]
	ro,co = row_col_order(df_beta)
	df_beta = df_beta.loc[ro,co]
	df_beta[df_beta>max_thresh] = max_thresh
	sns.clustermap(df_beta.T,cmap='viridis')
	plt.savefig(fname+'_bhmap'+'_th_'+str(max_thresh)+'.png');plt.close()
 
def plot_marker_genes(fn,df,umap_coords,marker_genes,nr,nc):

	from anndata import AnnData
	import scanpy as sc
	import numpy as np

	import matplotlib.pylab as plt
	plt.rcParams['figure.figsize'] = [15, 10]
	plt.rcParams['figure.autolayout'] = True
	import seaborn as sns

	adata = AnnData(df.to_numpy())
	sc.pp.normalize_total(adata, target_sum=1e4)
	sc.pp.log1p(adata)
	dfn = adata.to_df()
	dfn.columns = df.columns
	dfn['cell'] = df.index.values

	dfn['umap1']= umap_coords[:,0]
	dfn['umap2']= umap_coords[:,1]

	fig, ax = plt.subplots(nr,nc) 
	ax = ax.ravel()

	for i,g in enumerate(marker_genes):
		if g in dfn.columns:
			print(g)
			val = np.array([x if x<3 else 3.0 for x in dfn[g]])
			sns.scatterplot(data=dfn, x='umap1', y='umap2', hue=val,s=.1,palette="viridis",ax=ax[i],legend=False)

			# norm = plt.Normalize(val.min(), val.max())
			# sm = plt.cm.ScalarMappable(cmap="viridis",norm=norm)
			# sm.set_array([])

			# cax = fig.add_axes([ax[i].get_position().x1, ax[i].get_position().y0, 0.01, ax[i].get_position().height])
			# fig.colorbar(sm,ax=ax[i])
			# ax[i].axis('off')

			ax[i].set_title(g)
	fig.savefig(fn+'_umap_marker_genes_legend.png',dpi=600);plt.close()