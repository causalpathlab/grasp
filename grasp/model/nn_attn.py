import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import random 
from .loss import pcl_loss,pcl_loss_with_rare_cluster,pcl_loss_with_weighted_cluster,latent_alignment_loss
import logging
logger = logging.getLogger(__name__)
import numpy as np


import torch.nn.init as init
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class GRASPOUT:
    def __init__(self,h_c1,h_c2,z_c1,z_c2,attn_c1=None, attn_c2=None):
        self.h_c1 = h_c1
        self.h_c2 = h_c2
        self.z_c1 = z_c1
        self.z_c2 = z_c2
        self.attn_c1 = attn_c1
        self.attn_c2 = attn_c2

class GRASPel:
    def __init__(self,el_attn_c1,el_attn_c2, el_cl_c1,el_cl_c2):
        self.el_attn_c1 = el_attn_c1
        self.el_attn_c2 = el_attn_c2
        self.el_cl_c1 = el_cl_c1
        self.el_cl_c2 = el_cl_c2
                   
class Stacklayers(nn.Module):
    
    def __init__(self,
        input_size:int,
        layers:list,
        dropout:float=0.1
        ):
        super(Stacklayers, self).__init__()
        
        self.layers = nn.ModuleList()
        self.input_size = input_size
        for next_l in layers:
            self.layers.append(nn.Linear(self.input_size,next_l))
            self.layers.append(nn.BatchNorm1d(next_l))
            self.layers.append(self.get_activation())
            self.layers.append(nn.Dropout(dropout))
            self.input_size = next_l

    def forward(self, 
        input_data:torch.tensor):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

    def get_activation(self):
        return nn.ReLU()

class GeneEmbedor(nn.Module):
    
    def __init__(self,
        emb_dim:int,
        out_dim:int,
        ):
        super(GeneEmbedor, self).__init__()
        
        self.embedding = nn.Embedding(emb_dim,out_dim)
        self.emb_norm = nn.LayerNorm(out_dim)
        self.emb_dim = emb_dim

    def forward(self,
        x:torch.tensor):
        
        row_sums = x.sum(dim=1, keepdim=True)
        x_norm = torch.div(x, row_sums) * (self.emb_dim -1)
        return self.emb_norm(self.embedding(x_norm.int()))

class ScaledDotAttention(nn.Module):
    
    def __init__(self,
        weight_dim:int,
        input_dim:int,
        pair_importance_weight:float
        ):
        super(ScaledDotAttention, self).__init__()
        
        self.W_query = nn.Parameter(torch.randn(weight_dim, weight_dim))
        self.W_key = nn.Parameter(torch.randn(weight_dim, weight_dim))
        self.W_value = nn.Parameter(torch.randn(weight_dim, weight_dim))
        self.model_dim = weight_dim
        self.pair_importance_weight = pair_importance_weight
        
    def forward(self,
        query:torch.tensor, 
        key:torch.tensor, 
        value:torch.tensor
        ):

        query_proj = torch.matmul(query, self.W_query)
        key_proj = torch.matmul(key, self.W_key)
        value_proj = torch.matmul(value, self.W_value)
        
        scores = torch.matmul(query_proj, key_proj.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.model_dim).float())
                    
        diag_bias = torch.eye(scores.shape[1], dtype=scores.dtype, device=scores.device) * torch.max(scores)
        p_importance = self.pair_importance_weight * diag_bias        
        scores = scores + p_importance
    
        attention_weights = torch.softmax(scores, dim=-1)
        entropy_loss_attn = -torch.mean(torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1))

        output = torch.matmul(attention_weights, value_proj)

        return output, attention_weights, entropy_loss_attn

class AttentionPooling(nn.Module):

    def __init__(self, 
        model_dim:int
        ):
        super(AttentionPooling, self).__init__()
        
        self.weights = nn.Parameter(torch.randn(model_dim))  
    
    def forward(self, 
        attention_output:torch.tensor
        ):
        
        weights_softmax = torch.softmax(self.weights, dim=0)
        weighted_output = attention_output * weights_softmax.unsqueeze(0)
        pooled_output = torch.sum(weighted_output, dim=-1, keepdim=True)
        return pooled_output.squeeze(-1)

class ENCODER(nn.Module):
    def __init__(self,
        input_dims:int,
        layers:list
        ):
        super(ENCODER, self).__init__()
        self.fc = Stacklayers(input_dims,layers)
  
    def forward(self, x:torch.tensor):
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self,
        input_dims:int,
        layers:list
        ):
        super(MLP, self).__init__()
        
        self.fc = Stacklayers(input_dims,layers)

    def forward(self, x:torch.tensor):
        z = self.fc(x)
        return z

class GRASPNET(nn.Module):
    def __init__(self,
        input_dim:int, 
        embedding_dim:int, 
        attention_dim:int, 
        latent_dim:int,
        encoder_layers:list,
        projection_layers:list,
        corruption_tol:float,
        pair_importance_weight:float
        ):
        super(GRASPNET,self).__init__()

        self.embedding = GeneEmbedor(embedding_dim,attention_dim)
        self.attention = ScaledDotAttention(attention_dim,input_dim,pair_importance_weight)
        self.pooling = AttentionPooling(attention_dim)

        self.encoder = ENCODER(input_dim,encoder_layers)
        self.projector = MLP(latent_dim, projection_layers)
        
        self.corruption_tol = corruption_tol
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Parameter):
            init.xavier_uniform_(module)
            
    def forward(self,x_c1,x_c2,nbr_weight=None):
        
        if nbr_weight != None:
            mean_val = torch.mean(nbr_weight)
            std_val = torch.std(nbr_weight)
            threshold = self.corruption_tol * std_val
            outliers = torch.where(torch.abs(nbr_weight - mean_val) > threshold)
            outlier_indices = outliers[0]
            
            all_indices = torch.arange(nbr_weight.size(0))
            non_outlier_indices = torch.tensor([i for i in all_indices if i not in outlier_indices])
            sampled_indices = random.sample(non_outlier_indices.tolist(), len(outlier_indices))

            x_c1[outlier_indices] = x_c1[sampled_indices]
            x_c2[outlier_indices] = x_c2[sampled_indices]

        x_c1_emb = self.embedding(x_c1)
        x_c2_emb = self.embedding(x_c2)
  
        x_c1_att_out, x_c1_att_w,el_attn_c1 = self.attention(x_c1_emb,x_c2_emb,x_c2_emb)
        x_c1_pool_out = self.pooling(x_c1_att_out)

        x_c2_att_out, x_c2_att_w,el_attn_c2 = self.attention(x_c2_emb,x_c1_emb,x_c1_emb)
        x_c2_pool_out = self.pooling(x_c2_att_out)

        h_c1 = self.encoder(x_c1_pool_out)
        h_c2 = self.encoder(x_c2_pool_out)

        z_c1 = self.projector(h_c1)
        z_c2 = self.projector(h_c2)
                
        pred_c1 = torch.softmax(h_c1, dim=1)
        el_cl_c1 = -torch.mean(torch.sum(pred_c1 * torch.log(pred_c1 + 1e-10), dim=1))

        pred_c2 = torch.softmax(h_c2, dim=1)
        el_cl_c2 = -torch.mean(torch.sum(pred_c2 * torch.log(pred_c2 + 1e-10), dim=1))

        return GRASPOUT(h_c1,h_c2,z_c1,z_c2,x_c1_att_w,x_c2_att_w), GRASPel(el_attn_c1,el_attn_c2, el_cl_c1,el_cl_c2)

def train(model,data,
    epochs:int,
    lambda_loss:float,
    l_rate:float,
    cl_loss_mode:str, 
    loss_clusters:float, 
    loss_threshold:float, 
    loss_weight:float,
    temperature:float,
    min_batchsize:int
    ):
    
    logger.info('Init training....nn_attn')
    opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
    epoch_losses = []
    lambda_attn_loss = float(lambda_loss[0])
    lambda_latent_loss = float(lambda_loss[1])
    lambda_latalign_loss = float(lambda_loss[2])
    lambda_cl_loss = float(lambda_loss[3])
    for epoch in range(epochs):
        epoch_l, cl, el, el_attn_c1, el_attn_c2, el_cl_c1, el_cl_c2 = (0,) * 7
        for x_c1,y,x_c2,nbr_weight in data:
                        
            if x_c1.shape[0] < min_batchsize:
                continue
            
            opt.zero_grad()

            GRASP_out,GRASP_el = model(x_c1,x_c2,nbr_weight)

            if cl_loss_mode == 'rare':
                cl_loss = lambda_cl_loss * pcl_loss_with_rare_cluster(GRASP_out.z_c1, GRASP_out.z_c2,loss_clusters, loss_threshold, loss_weight,temperature)
            elif cl_loss_mode == 'weighted':
                cl_loss = lambda_cl_loss * pcl_loss_with_weighted_cluster(GRASP_out.z_c1, GRASP_out.z_c2,loss_clusters,loss_weight,temperature)
            else:
                cl_loss = lambda_cl_loss * pcl_loss(GRASP_out.z_c1, GRASP_out.z_c2,temperature)


            entropy_loss = (GRASP_el.el_attn_c1 * lambda_attn_loss +
                        GRASP_el.el_attn_c2 * lambda_attn_loss +
                        GRASP_el.el_cl_c1 * lambda_latent_loss +
                        GRASP_el.el_cl_c2 * lambda_latent_loss)
            
            alignment_loss = latent_alignment_loss(GRASP_out.z_c1, GRASP_out.z_c2) * lambda_latalign_loss

            train_loss = cl_loss + entropy_loss + alignment_loss
            
            train_loss.backward()

            opt.step()
            epoch_l += train_loss.item()
            cl += cl_loss.item() + alignment_loss.item()
            el += entropy_loss.item()
            el_attn_c1 += GRASP_el.el_attn_c1.item()
            el_attn_c2 += GRASP_el.el_attn_c2.item()
            el_cl_c1 += GRASP_el.el_cl_c1.item()
            el_cl_c2 += GRASP_el.el_cl_c2.item()
           
        epoch_losses.append([epoch_l/len(data),cl/len(data),el/len(data),el_attn_c1/len(data),el_attn_c2/len(data),el_cl_c1/len(data),el_cl_c2/len(data)])  
        
        if epoch % 10 == 0:
            logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch,epoch_l/len(data) ))

        return epoch_losses
 
def predict_batch(model,x_c1,y,x_c2 ):
    return model(x_c1,x_c2),y

def predict_attention(model,
    x_c1:torch.tensor,
    x_c2:torch.tensor
    ):
    
    x_c1_emb = model.embedding(x_c1)
    x_c2_emb = model.embedding(x_c2)

    _,x_c1_attention,_ = model.attention(x_c1_emb,x_c2_emb,x_c2_emb)
    
    return x_c1_attention

def predict_context(model,
    x_c1:torch.tensor,
    x_c2:torch.tensor                
    ):
    
    x_c1_emb = model.embedding(x_c1)
    x_c2_emb = model.embedding(x_c2)

    x_c1_context,_,_ = model.attention(x_c1_emb,x_c2_emb,x_c2_emb)
    
    return x_c1_context

def get_latent(model,
    x_c1:torch.tensor,
    x_c2:torch.tensor
    ):
    
    x_c1_emb = model.embedding(x_c1)
    x_c2_emb = model.embedding(x_c2)
    x_c1_context,_,_ = model.attention(x_c1_emb,x_c2_emb,x_c2_emb)
    x_c1_pool_out = model.pooling(x_c1_context)
    h_c1 = model.encoder(x_c1_pool_out)
    return h_c1
    