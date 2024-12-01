"""
modifiled from welch-lab/pyliger
"""

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from numpy.typing import NDArray
from sklearn.cluster import KMeans
import leidenalg
import igraph as ig
from scipy.sparse import csr_matrix
import logging


def run_ann(mtx,k, num_trees=None,dist=False):

	num_observations = mtx.shape[0]
	# decide number of trees
	if num_trees is None:
		if num_observations < 100000:
			num_trees = 10
		elif num_observations < 1000000:
			num_trees = 20
		elif num_observations < 5000000:
			num_trees = 50
		else:
			num_trees = 100

	# build knn graph
	t = AnnoyIndex(mtx.shape[1], 'angular')
	for i in range(num_observations):
		t.add_item(i, mtx[i])
	t.build(num_trees)

	if dist:
	
		knn_idx =[] 
		knn_dist = []
		
		for i in range(mtx.shape[0]):
			ki,kd = t.get_nns_by_vector(mtx[i], k,include_distances=True)
			knn_idx.append(ki)
			knn_dist.append(kd)  
					  
		return np.asarray(knn_idx), np.asarray(knn_dist)
	
	else:
		
		mtx_knn = np.vstack([t.get_nns_by_vector(mtx[i], k) for i in range(num_observations)])
		return mtx_knn

def umap_connectivity(
	knn_indices: NDArray[np.int32],
	knn_dists: NDArray[np.float32],
	*,
	n_obs: int,
	n_neighbors: int,
	set_op_mix_ratio: float = 1.0,
	local_connectivity: float = 1.0,
) -> csr_matrix:
	"""
	from scanpy/umap module
	"""
	from umap.umap_ import fuzzy_simplicial_set
	from scipy.sparse import coo_matrix
	
	X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
	connectivities = fuzzy_simplicial_set(
		X,
		n_neighbors,
		None,
		None,
		knn_indices=knn_indices,
		knn_dists=knn_dists,
		set_op_mix_ratio=set_op_mix_ratio,
		local_connectivity=local_connectivity,
	)

	if isinstance(connectivities, tuple):
		# In umap-learn 0.4, this returns (result, sigmas, rhos)
		connectivities = connectivities[0]

	return connectivities.tocsr()


def build_igraph(snn):
	sources, targets = snn.nonzero()
	weights = snn[sources, targets]

	if isinstance(weights, np.matrix):
		weights = weights.A1
	g = ig.Graph()
	g.add_vertices(snn.shape[0])
	g.add_edges(list(zip(sources, targets)))
	g.es['weight'] = weights

	return g

def compute_snn(knn, prune):
	"""helper function to compute the SNN graph
	
	https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6280782/
	
	"""
	# int for indexing
	knn = knn.astype(np.int32)

	k = knn.shape[1]
	num_cells = knn.shape[0]

	rows = np.repeat(list(range(num_cells)), k)
	columns = knn.flatten()
	data = np.repeat(1, num_cells * k)
	snn = csr_matrix((data, (rows, columns)), shape=(num_cells, num_cells))

	snn = snn @ snn.transpose()

	rows, columns = snn.nonzero()
	data = snn.data / (k + (k - snn.data))
	data[data < prune] = 0

	return csr_matrix((data, (rows, columns)), shape=(num_cells, num_cells))

def leiden_cluster(mtx,
				   resolution=1.0,
				   k=15,
				   prune=1/15,
				   random_seed=1,
				   n_iterations=-1,
				   n_starts=10,
				   method = 'fuzzy_conn',
				   center = True):

	logging.info('Running leiden cluster....')
	logging.info('resolution: '+str(resolution))
	logging.info('k: '+str(k))
	logging.info('method: '+str(method))
	
	if center:
		mtx = (mtx - np.mean(mtx, axis=0)) / np.std(mtx, axis=0, ddof=1)

	if method == 'shared_nn':

		knn = run_ann(mtx,k)

		snn = compute_snn(knn, prune=prune)

		g = build_igraph(snn)

		np.random.seed(random_seed)
		max_quality = -1
		for i in range(n_starts):  
			seed = np.random.randint(0, 1000)
			kwargs = {'weights': g.es['weight'], 'resolution_parameter': resolution, 'seed': seed}  
			part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, n_iterations=n_iterations, **kwargs)

			if part.quality() > max_quality:
				cluster = part.membership
				max_quality = part.quality()

		return snn, cluster

	elif method == 'fuzzy_conn':
		
		knn_idx, knn_dist = run_ann(mtx,k=k,dist=True)
		connectivities = umap_connectivity(
				knn_idx,
				knn_dist,
				n_obs=mtx.shape[0],
				n_neighbors=k,
			)

		g = build_igraph(connectivities)

		np.random.seed(random_seed)
		max_quality = -1
		for i in range(n_starts):  
			seed = np.random.randint(0, 1000)
			kwargs = {'weights': g.es['weight'], 'resolution_parameter': resolution, 'seed': seed}  
			part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, n_iterations=n_iterations, **kwargs)

			if part.quality() > max_quality:
				cluster = part.membership
				max_quality = part.quality()

		return connectivities, cluster

def kmeans_cluster(mtx,
                   k,
				   random_seed=1,
				   center = True):

	logging.info('Running kmeans cluster....')
	logging.info('k: '+str(k))
	
	if center:
		mtx = (mtx - np.mean(mtx, axis=0)) / np.std(mtx, axis=0, ddof=1)

		
	kmeans = KMeans(n_clusters=k, init='k-means++',random_state=random_seed).fit(mtx)
	cluster = kmeans.labels_

	return cluster
