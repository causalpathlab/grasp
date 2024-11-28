import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scvi.distributions import ZeroInflatedNegativeBinomial


def reparameterize(mean,lnvar):
	sig = torch.exp(lnvar/2.)
	eps = torch.randn_like(sig)
	return eps.mul_(sig).add_(mean)

def multinm_log_likelihood(xx,pr, eps=1e-8):
	return torch.sum(xx * torch.log(pr+eps),dim=-1)

def multi_dir_log_likelihood(x,alpha):
	a = torch.lgamma(alpha.sum(1)) - torch.lgamma(alpha).sum(1)
	b = torch.lgamma(x + alpha).sum(1) - torch.lgamma( (x + alpha).sum(1))
	return a + b 

def kl_loss(mean,lnvar):
	return  -0.5 * torch.sum(1. + lnvar - torch.pow(mean,2) - torch.exp(lnvar), dim=-1)

def get_zinb_reconstruction_loss(x, px_s, px_r, px_d):
	'''https://github.com/scverse/scvi-tools/blob/master/scvi/module/_vae.py'''
	return torch.mean(-ZeroInflatedNegativeBinomial(mu=px_s, theta=px_r, zi_logits=px_d).log_prob(x).sum(dim=-1))

def minimal_overlap_loss(z_common, z_unique):
	z_common_norm = F.normalize(z_common, p=2, dim=-1)
	z_unique_norm = F.normalize(z_unique, p=2, dim=-1)
	cosine_similarity = torch.sum(z_common_norm * z_unique_norm, dim=-1)
	return torch.mean(torch.abs(cosine_similarity))


def cl_loss(anchor, positive, negative, margin=1.0):
	# distance_positive = (anchor - positive).pow(2).sum(1)  
	# distance_negative = (anchor - negative).pow(2).sum(1)  

	distance_positive = F.pairwise_distance(anchor, positive, keepdim=True)
	distance_negative = F.pairwise_distance(anchor, negative, keepdim=True)

	losses = F.relu(distance_positive - distance_negative + margin)

	return losses.mean()

def identify_rare_groups(latent_space, num_clusters, rare_group_threshold):
	kmeans = KMeans(n_clusters=num_clusters)
	cluster_labels = kmeans.fit_predict(latent_space.cpu().detach().numpy())
	
	cluster_counts = torch.tensor([(cluster_labels == i).sum() for i in range(num_clusters)])
	total_samples = len(cluster_labels)
	rare_clusters = cluster_counts < (rare_group_threshold * total_samples)
	
	return torch.tensor(cluster_labels, device=latent_space.device), rare_clusters

def pcl_loss_with_rare_cluster(z_i, z_j, num_clusters, rare_group_threshold, rare_group_weight,temperature=1.0):
	batch_size = z_i.size(0)

	z = torch.cat([z_i, z_j], dim=0)
	similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

	sim_ij = torch.diag(similarity, batch_size)
	sim_ji = torch.diag(similarity, -batch_size)
	positives = torch.cat([sim_ij, sim_ji], dim=0)

	mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
	numerator = torch.exp(positives / temperature)
	denominator = mask * torch.exp(similarity / temperature)

	all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))

	cluster_labels, rare_clusters = identify_rare_groups(z, num_clusters, rare_group_threshold)
	
	weights = torch.ones_like(all_losses, device=z_i.device)
	for cluster_idx in range(num_clusters):
		if rare_clusters[cluster_idx]:
			cluster_indices = (cluster_labels == cluster_idx).nonzero(as_tuple=True)[0]
			weights[cluster_indices] = rare_group_weight
	weighted_losses = all_losses * weights
	loss = torch.sum(weighted_losses) / (2 * batch_size)

	return loss

def pcl_loss_with_weighted_cluster(z_i, z_j, num_clusters, unmatched_group_weight,temperature=1.0):

	batch_size = z_i.size(0)
	
	z = torch.cat([z_i, z_j], dim=0)
	
	z_np = z.detach().cpu().numpy()
	kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(z_np)
	cluster_assignments = kmeans.labels_

	similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

	sim_ij = torch.diag(similarity, batch_size)
	sim_ji = torch.diag(similarity, -batch_size)
	positives = torch.cat([sim_ij, sim_ji], dim=0)

	mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()

	weight_matrix = torch.ones_like(similarity, device=z.device)

	for i in range(batch_size * 2):
		for j in range(batch_size * 2):
			if i != j:  
				if cluster_assignments[i] != cluster_assignments[j]:
					weight_matrix[i, j] += unmatched_group_weight
	  

	numerator = torch.exp(positives / temperature)
	denominator = mask * torch.exp(similarity / temperature) * weight_matrix

	all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
	loss = torch.sum(all_losses) / (2 * batch_size)

	return loss



def pcl_loss(z_i, z_j,temperature = 1.0):
	
		batch_size = z_i.size(0)

		# compute similarity between the sample's embedding and its corrupted view
		z = torch.cat([z_i, z_j], dim=0)
		similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

		sim_ij = torch.diag(similarity, batch_size)
		sim_ji = torch.diag(similarity, -batch_size)
		positives = torch.cat([sim_ij, sim_ji], dim=0)

		mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
		numerator = torch.exp(positives / temperature)
		denominator = mask * torch.exp(similarity / temperature)

		all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
		loss = torch.sum(all_losses) / (2 * batch_size)

		return loss
	
def tcl_loss(z_a, z_p, z_n,temperature = 1.0):
	dist_pos = (z_a - z_p).pow(2).sum(1)
	dist_neg = (z_a - z_n).pow(2).sum(1)
	loss = F.relu(dist_pos - dist_neg + temperature)
	return loss.mean()

def tcl_ce_loss(z_a, z_p, z_n, temperature = 1.0):
	
	batch_size = z_a.size(0)

	pos_similarity = F.cosine_similarity(z_a, z_p, dim=-1)
	neg_similarity = F.cosine_similarity(z_a.unsqueeze(1), z_n.unsqueeze(0), dim=2)

	numerator = torch.exp(pos_similarity/temperature)
	denominator = torch.exp(pos_similarity / temperature) + torch.sum(torch.exp(neg_similarity / temperature), dim=1)

	mask = torch.eye(batch_size, dtype=torch.bool, device=z_a.device)
	denominator = denominator.masked_fill(mask, 0)

	all_losses = -torch.log(numerator / denominator)
	loss = torch.mean(all_losses)
	
	# print("NaN values in :", torch.isnan(pos_similarity).any())
	
	return loss

def latent_alignment_loss(z_c1, z_c2):
	return 1 - F.cosine_similarity(z_c1, z_c2).mean()