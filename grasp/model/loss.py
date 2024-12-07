import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scvi.distributions import ZeroInflatedNegativeBinomial


def get_zinb_reconstruction_loss(x, px_s, px_r, px_d):
	'''https://github.com/scverse/scvi-tools/blob/master/scvi/module/_vae.py'''
	return torch.mean(-ZeroInflatedNegativeBinomial(mu=px_s, theta=px_r, zi_logits=px_d).log_prob(x).sum(dim=-1))

def minimize_similarity(z_1, z_2):
	z_1_norm = F.normalize(z_1, p=2, dim=-1)
	z_2_norm = F.normalize(z_2, p=2, dim=-1)
	cosine_similarity = torch.sum(z_1_norm * z_2_norm, dim=-1)
	return torch.mean(torch.abs(cosine_similarity))


def minimal_overlap_loss(z_1, z_2):

	# decrease similarity between latent spaces
    alignment_loss = minimize_similarity(z_1, z_2)
        
    total_loss = alignment_loss

    return total_loss
