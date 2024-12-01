import logging
import numpy as np
import numba as nb
import pandas as pd
from math import sqrt
logger = logging.getLogger(__name__)


@nb.njit(parallel=True)
def calc_res(mtx,sum_gene,sum_cell,sum_total,theta,clip,n_gene,n_cell):
    
    def clac_clipped_res_dense(gene: int, cell: int) -> np.float64:
        mu = sum_gene[gene] * sum_cell[cell] / sum_total
        value = mtx[cell, gene]

        mu_sum = value - mu
        pre_res = mu_sum / sqrt(mu + mu * mu / theta)
        res = np.float64(min(max(pre_res, -clip), clip))
        return res

    norm_gene_var = np.zeros(n_gene, dtype=np.float64)

    for gene in nb.prange(n_gene):
        sum_clipped_res = np.float64(0.0)
        for cell in range(n_cell):
            sum_clipped_res += clac_clipped_res_dense(gene, cell)
        mean_clipped_res = sum_clipped_res / n_cell

        var_sum = np.float64(0.0)
        for cell in range(n_cell):
            clipped_res = clac_clipped_res_dense(gene, cell)
            diff = clipped_res - mean_clipped_res
            var_sum += diff * diff

        norm_gene_var[gene] = var_sum / n_cell

    return norm_gene_var

def select_hvgenes(mtx,gene_var_z):
    '''
    adapted from pyliger plus scanpy's seurat high variable gene selection

    '''
                            
    sum_gene = np.array(mtx.sum(axis=0)).ravel()
    sum_cell = np.array(mtx.sum(axis=1)).ravel()
    sum_total = np.float64(np.sum(sum_gene).ravel())
    n_gene = mtx.shape[1]
    n_cell = mtx.shape[0]
    
    theta = np.float64(100)
    clip = np.float64(np.sqrt(n_cell))
    norm_gene_var = calc_res(mtx,sum_gene,sum_cell,sum_total,theta,clip,n_gene,n_cell)
    norm_gene_var = np.nan_to_num(norm_gene_var)
    select_genes = norm_gene_var>gene_var_z

    return select_genes

def select_hvgenes_scanpy(mtx,n_genes=2000):
    import scanpy as sc
    adata = sc.AnnData(mtx)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata,n_top_genes=n_genes)
    return adata.var['highly_variable'].values