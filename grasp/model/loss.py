import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scvi.distributions import ZeroInflatedNegativeBinomial


def get_zinb_reconstruction_loss(x, px_s, px_r, px_d):
	'''https://github.com/scverse/scvi-tools/blob/master/scvi/module/_vae.py'''
	return torch.mean(-ZeroInflatedNegativeBinomial(mu=px_s, theta=px_r, zi_logits=px_d).log_prob(x).sum(dim=-1))

def minimal_overlap_loss(z_common, z_unique):
	z_common_norm = F.normalize(z_common, p=2, dim=-1)
	z_unique_norm = F.normalize(z_unique, p=2, dim=-1)
	cosine_similarity = torch.sum(z_common_norm * z_unique_norm, dim=-1)
	return torch.mean(torch.abs(cosine_similarity))

