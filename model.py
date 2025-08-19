import os
import csv
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, densenet121

from torch.nn import LayerNorm, MultiheadAttention
from torch_geometric.nn import SAGEConv, Set2Set


from utils import (
    load_ST_file,
    calculate_adj_matrix,
    refine,
    build_her2st_data,
    get_predicted_results,
)
from metrics import eval_mclust_ari
from loss import NT_Xent, DCL, DCLW


    
class GraphSAGE(nn.Module):
    """
    Two-layer GraphSAGE encoder
    """

    def __init__(self, num_node_features, hidden_dim, last_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_dim)  
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, last_dim)           
        self.bn2 = nn.LayerNorm(last_dim)

    def forward(self, data):
        """
        Forward pass on a PyG Data object
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class AttentionLayer(nn.Module):
    """
    Attention pooling across embeddings 
    """
    def __init__(self, embedding_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Linear(embedding_dim, 1)   # score per embedding
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings):
        """
        return: 
        (aggregated [batch_size, dim], attention_weights [batch_size, num_embeddings])
        """
        embeddings = self.layer_norm(embeddings)
        attention_scores = self.attention_fc(embeddings)              
        attention_weights = F.softmax(attention_scores, dim=1)        
        weighted_embeddings = embeddings * attention_weights          
        aggregated_embeddings = torch.sum(weighted_embeddings, dim=1) 
        return aggregated_embeddings, attention_weights.squeeze(-1)


def LinearBlock(input_dim, output_dim, p_drop):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ELU(),
        nn.Dropout(p=p_drop)
    )


class PathCLAST(nn.Module):
    """
    Spatial contrastive model for gene graphs and histology images
    """

    def __init__(self, num_node_features, hidden_dim, last_dim, image_dims, p_drop, n_pos,
                 backbone='densenet', projection_dims=[64, 64]):
        super(PathCLAST, self).__init__()

        # gene pathway encoder
        self.gene_encoder = GraphSAGE(num_node_features, hidden_dim, last_dim)
        self.set2set = Set2Set(last_dim, processing_steps=3)              # graph-level readout
        self.set2set_linear = nn.Linear(2 * last_dim, last_dim)           # reduce 2*D → D
        self.attention_layer = AttentionLayer(last_dim)                   # pathway-wise attention

        # image backbone
        if backbone == 'densenet':
            self.image_encoder = densenet121(pretrained=True)
            n_features = self.image_encoder.classifier.in_features
            self.image_encoder.classifier = nn.Identity()                  # remove classifier head
        elif backbone == 'resnet':
            self.image_encoder = resnet50(pretrained=True)
            n_features = self.image_encoder.fc.in_features
            self.image_encoder.fc = nn.Identity()                          # remove classifier head

        # positional embeddings for (x, y)
        self.x_embedding = nn.Embedding(n_pos, n_features)
        self.y_embedding = nn.Embedding(n_pos, n_features)

        # image MLP
        image_dims[0] = n_features
        image_dims.append(projection_dims[0])
        self.image_linear = nn.Sequential(OrderedDict([
            (f'image_block{i+1}', LinearBlock(image_dims[i], image_dims[i+1], p_drop))
            for i, _ in enumerate(image_dims[:-1])
        ]))

        # projector
        self.projector = nn.Sequential(
            nn.Linear(projection_dims[0], projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1]),
        )

    def forward_image(self, xi, spatial):
        """
        Encode image and project to contrastive space
        return: (pre-projection features, projected features)
        """
        xi = self.image_encoder(xi)           
        xi = self.image_linear(xi)            
        hi = self.projector(xi)               
        return xi, hi

    def forward_gene(self, xg_list):
        """
        Encode pathway graphs, aggregate with attention, and project.
        return: (pre-proj aggregated, projected, attention weights)
        """
        all_embeddings = []
        for xg_batch in xg_list:
            embeddings = self.gene_encoder(xg_batch)                  # node embeddings
            graph_embeddings = self.set2set(embeddings, xg_batch.batch)  # graph readout
            graph_embeddings = self.set2set_linear(graph_embeddings)  # D
            all_embeddings.append(graph_embeddings)

        all_embeddings = torch.stack(all_embeddings)                 
        aggregated_embeddings, attention_weights = self.attention_layer(all_embeddings)
        hg = self.projector(aggregated_embeddings)                   
        return aggregated_embeddings, hg, attention_weights

    def forward(self, xg_list, xi, spatial):
        """
        Full forward for pathway graphs and image
        return: 
        xg(pre-proj), xi(pre-proj), hg, hi, pathway_attn_weights
        """
        xg, hg, pathway_attn_weights = self.forward_gene(xg_list)
        xi, hi = self.forward_image(xi, spatial)
        return xg, xi, hg, hi, pathway_attn_weights


class TrainerPathCLAST:
    def __init__(self, args, n_clusters, network, optimizer, log_dir, device='cuda'):
        self.n_clusters = n_clusters
        self.network = network
        self.optimizer = optimizer
        
        self.train_writer = SummaryWriter(log_dir+'_train')
        self.valid_writer = SummaryWriter(log_dir+'_valid')
        self.device = device

        self.args = args
        if args.dataset == "DLPFC":
            adata = load_ST_file(os.path.join(args.path, args.name))
            df_meta = pd.read_csv(os.path.join(args.path, args.name, 'metadata.tsv'), sep='\t')
            label = pd.Categorical(df_meta['ground_truth']).codes
            adata = adata[label != -1]
            self.sample_id = adata.obs.index.tolist()
            self.adj_2d = calculate_adj_matrix(x=adata.obs["array_row"].tolist(), y=adata.obs["array_col"].tolist(), histology=False)
        elif args.dataset == "Her2st":
            adata, _ = build_her2st_data(args.path, args.name, args.img_size)
            label = adata.obs['label']
            adata = adata[label != -1]
            self.sample_id = adata.obs.index.tolist()
            self.adj_2d = calculate_adj_matrix(x=adata.obsm["spatial"][:, 0].tolist(), y=adata.obsm["spatial"][:, 1].tolist(), histology=False)
        elif args.dataset=="IDC":
            adata = load_ST_file(os.path.join(args.path, args.name))
            df_meta = pd.read_csv(os.path.join(args.path, args.name, 'metadata.tsv'), sep='\t')
            label = pd.Categorical(df_meta['ground_truth']).codes
            n_clusters = label.max() + 1
            adata = adata[label != -1]
            self.sample_id = adata.obs.index.tolist()
            self.adj_2d = calculate_adj_matrix(x=adata.obs["array_row"].tolist(), y=adata.obs["array_col"].tolist(),
                                      histology=False)
        
        self.w_g2g = args.w_g2g
        self.w_i2i = args.w_i2i
        
    def eval_mclust_refined_ari(self, label, z):
        if z.shape[0] < 1000:
            print('z shape 0 : ', z.shape[0])
            num_nbs = 4
        else:
            num_nbs = 24
        ari, preds = eval_mclust_ari(label, z, self.n_clusters)
        refined_preds = refine(sample_id=self.sample_id, pred=preds, dis=self.adj_2d, num_nbs=num_nbs)
        ari = adjusted_rand_score(label, refined_preds)
        return ari

    def train(self, trainloader, epoch):
        """
        One epoch training step
        """

        self.network.train()
        train_loss = 0.0
        train_cnt = 0

        with tqdm(total=len(trainloader), desc=f"Epoch {epoch} train") as t:
            for batch in trainloader:
                self.optimizer.zero_grad()

                xg, xg_u, xg_v, xi_u, xi_v, spatial, y, _ = batch
                xg = [g.to(self.device) for g in xg]
                xg_u = [g.to(self.device) for g in xg_u]
                xg_v = [g.to(self.device) for g in xg_v]
                xi_u, xi_v = xi_u.to(self.device), xi_v.to(self.device)
                spatial = spatial.to(self.device)

                _, hg, _ = self.network.forward_gene(xg)
                _, hg_u, _ = self.network.forward_gene(xg_u)
                _, hg_v, _ = self.network.forward_gene(xg_v)
                _, hi_u = self.network.forward_image(xi_u, spatial)
                _, hi_v = self.network.forward_image(xi_v, spatial)

                # losses
                criterion = NT_Xent(hg.shape[0])
                g2g_loss = criterion(hg_u, hg_v) * self.w_g2g
                i2i_loss = criterion(hi_u, hi_v) * self.w_i2i
                g2i_loss = criterion(hg, hi_u)

                loss = g2i_loss + g2g_loss + i2i_loss
                loss.backward()
                self.optimizer.step()

                train_cnt += 1
                train_loss += loss.item()
                avg_train_loss = train_loss / train_cnt

                t.set_postfix(
                    loss=f"{avg_train_loss:.3f}",
                    g2i_loss=f"{g2i_loss.item():.3f}",
                    g2g_loss=f"{g2g_loss.item():.3f}",
                    i2i_loss=f"{i2i_loss.item():.3f}"
                )
                t.update(1)

        avg_train_loss = train_loss / train_cnt
        self.train_writer.add_scalar("loss", avg_train_loss, epoch)
        self.train_writer.flush()

        return avg_train_loss
            
    def valid(self, validloader, epoch=0):
        Xg = []
        Xi = []
        Y = []

        with torch.no_grad():
            with tqdm(total=len(validloader)) as t:
                self.network.eval()
                valid_loss = 0
                valid_cnt = 0

                for i, batch in enumerate(validloader):
                    xg, xi, spatial, y, _ = batch
                    xg = [item.to(self.device) for item in xg]
                    xi = xi.to(self.device)
                    spatial = spatial.to(self.device)
    
                 
                    xg, xi, hg, hi, _ = self.network(xg, xi, spatial)
                    criterion = NT_Xent(xg.shape[0])
                    loss = criterion(hg, hi)

                    valid_cnt += 1
                    valid_loss += loss.item()

                    Xg.append(xg.detach().cpu().numpy())
                    Xi.append(xi.detach().cpu().numpy())
                    Y.append(y)
                    t.set_postfix(loss=f'{(valid_loss / valid_cnt):.3f}')
                    t.update(1)

                Xg = np.vstack(Xg)
                Xi = np.vstack(Xi)
                Y = np.concatenate(Y, 0)
 
        return Xg, Xi, Y, valid_loss / valid_cnt
    
    def fit(self, trainloader, epochs):
        self.network = self.network.to(self.device)
        for epoch in range(epochs):
            self.train(trainloader, epoch + 1)


    def get_embeddings(self, validloader, save_name):
        xg, xi, _, _ = self.valid(validloader)
        np.save(os.path.join('preds', f'{save_name}_xg.npy'), xg)
        np.save(os.path.join('preds', f'{save_name}_xi.npy'), xi)

    def encode(self, batch):
        xg, xi, spatial, y, _ = batch
        xg = [item.to(self.device) for item in xg]
        xi = xi.to(self.device)
        spatial = spatial.to(self.device)
        xg, xi, hg, hi, _ = self.network(xg, xi, spatial)
        return xg +  xi

    def save_model(self, ckpt_path):
        torch.save(self.network.state_dict(), ckpt_path)

    def load_model(self, ckpt_path):
        self.network.load_state_dict(torch.load(ckpt_path))