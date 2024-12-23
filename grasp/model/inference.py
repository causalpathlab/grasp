import torch
import anndata as an
import numpy as np
import logging
logger = logging.getLogger(__name__)


def predict_batch_base(model,x_c1,y):
	return model(x_c1),y

def predict_batch_unique_gnn(model,x_c1,x_be_edge_index,x_ge_edge_index):
	return model(x_c1,x_be_edge_index,x_ge_edge_index)
