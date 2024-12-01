import pandas as pd
import numpy as np
import logging

def generate_gene_vals(df,top_n,top_genes,label):

	top_genes_collection = []
	for x in range(df.shape[0]):
		gtab = df.T.iloc[:,x].sort_values(ascending=False)[:top_n].reset_index()
		gtab.columns = ['gene','val']
		genes = gtab['gene'].values
		for g in genes:
			if g not in top_genes_collection:
				top_genes_collection.append(g)

	for g in top_genes_collection:
		for i,x in enumerate(df[g].values):
			top_genes.append(['k'+str(i),label,'g'+str(i+1),g,x])

	return top_genes

def get_topic_top_genes(df_beta,top_n):

	top_genes = []
	top_genes = generate_gene_vals(df_beta,top_n,top_genes,'top_genes')

	return pd.DataFrame(top_genes,columns=['Topic','GeneType','Genes','Gene','Proportion'])


def row_col_order(dfm):

	from scipy.cluster import hierarchy

	df = dfm.copy()
 
	Z = hierarchy.ward(df.to_numpy())
	ro = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, df.to_numpy()))

	df['topic'] = df.index.values
	dfm = pd.melt(df,id_vars='topic')
	dfm.columns = ['row','col','values']
	M = dfm[['row', 'col', 'values']].copy()
	M['row'] = pd.Categorical(M['row'], categories=ro)
	M = M.pivot(index='row', columns='col', values='values').fillna(0)
	co = np.argsort(-M.values.max(axis=0))
	co = M.columns[co]
 
	return ro,co

def quantile_normalization(ds1,ds2,	do_center=True, refine_knn=True, max_sample=1000, min_cells=20,quantiles=50):
	'''
	modified from pyliger package
	'''
	from scipy.stats.mstats import mquantiles
	from scipy import interpolate
	from ..clustering.leiden import refine_clusts
 
	def _mean_ties(x, y):
		"""helper function to calculate the mean value of y where ties(zeros) occur in x"""
		idx_zeros = x == 0
		if np.sum(idx_zeros) > 0:
			y[idx_zeros] = np.mean(y[idx_zeros])
		return y   

	adata_list = [ds1.copy(),ds2.copy()]
	num_samples = len(adata_list)
	ns = [adata.shape[0] for adata in adata_list]
	ref_dataset_idx = np.argmax(ns)
	use_these_factors = list(range(adata_list[0].shape[1]))
	num_clusters = len(use_these_factors)

	clusters = []
	for i in range(num_samples):
		if do_center:
			scale_H = (adata_list[i] - np.mean(adata_list[i], axis=0)) / np.std(adata_list[i], axis=0, ddof=1)
		else:
			scale_H = adata_list[i] / (
				np.sqrt(np.sum(np.square(adata_list[i]), axis=0) / (adata_list[i].shape[0] - 1))
			)

		clusts = np.argmax(scale_H, axis=1)

		if refine_knn:
			clusts = refine_clusts(adata_list[i], clusts, k=15)

		clusters.append(clusts)

	### Perform quantile alignment
	for k in range(num_samples):
		for j in range(num_clusters):
			cells2 = clusters[k] == j
			cells1 = clusters[ref_dataset_idx] == j

			for i in use_these_factors:
				num_cells2 = np.sum(cells2)
				num_cells1 = np.sum(cells1)

				if num_cells1 < min_cells or num_cells2 < min_cells:
					continue

				if num_cells2 == 1:
					adata_list[k][cells2, i] = np.mean(adata_list[ref_dataset_idx][cells1, i])
					continue

				q2 = mquantiles(
					np.random.permutation(adata_list[k][cells2, i])[
						0 : min(num_cells2, max_sample)
					],
					np.linspace(0, 1, num=quantiles + 1),
					alphap=1,
					betap=1,
				)
				max_H = np.max(adata_list[k][cells2, i])
				min_H = np.min(adata_list[k][cells2, i])
				if q2[-1] < max_H:
					q2[-1] = max_H
				if q2[0] > min_H:
					q2[0] = min_H
				q1 = mquantiles(
					np.random.permutation(adata_list[ref_dataset_idx][cells1, i])[
						0 : min(num_cells1, max_sample)
					],
					np.linspace(0, 1, num=quantiles + 1),
					alphap=1,
					betap=1,
				)

				if (
					np.sum(q1) == 0
					or np.sum(q2) == 0
					or len(np.unique(q1)) < 2
					or len(np.unique(q2)) < 2
				):
					new_vals = np.repeat(0, num_cells2)
				else:
					q1 = _mean_ties(q2, q1)
					warp_func = interpolate.interp1d(q2, q1)
					new_vals = warp_func(adata_list[k][cells2, i])
					adata_list[k][cells2, i] = new_vals

	return adata_list[0],adata_list[1]


def run_umap(mtx,
			 snn_graph = None,
		 	 use_snn = True,
			 k=2,
			 distance="euclidean",
			 n_neighbors=15,
			 min_dist=0.1,
			 rand_seed=42):
    
	logging.info('Running umap embedding....')
	logging.info('min_dist: '+str(min_dist))
	logging.info('n_neighbors: '+str(n_neighbors))
	logging.info('distance: '+str(distance))
	
	if use_snn:

		from umap.umap_ import find_ab_params, simplicial_set_embedding
		
		n_components = k
		spread: float = 1.0
		alpha: float = 1.0
		gamma: float = 1.0
		negative_sample_rate: int = 5
		maxiter = None
		default_epochs = 500 if snn_graph.shape[0] <= 10000 else 200
		n_epochs = default_epochs if maxiter is None else maxiter
		random_state = np.random.RandomState(rand_seed)

		a, b = find_ab_params(spread, min_dist)

		umap_coords = simplicial_set_embedding(
			data = mtx,
			graph = snn_graph,
			n_components=n_components,
			initial_alpha = alpha,
			a = a,
			b = b,
			gamma = gamma,
			negative_sample_rate = negative_sample_rate,
			n_epochs = n_epochs,
			init='spectral',
			random_state = random_state,
			metric = distance,
			metric_kwds = {},
			densmap=False,
			densmap_kwds={},
			output_dens=False
			)
		return umap_coords[0]

	else:
		import umap
		
		umap_coords = umap.UMAP(n_components=k, metric=distance,
							n_neighbors=n_neighbors, min_dist=min_dist,
							random_state=rand_seed).fit_transform(mtx)

		return umap_coords	
