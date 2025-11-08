import os
import csv
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import Set2Set
from torch_geometric.data import Batch as GeoBatch

import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, densenet121

from utils import load_ST_file, calculate_adj_matrix, refine, build_her2st_data, get_predicted_results
from metrics import eval_mclust_ari
from loss import NT_Xent
    
class GraphSAGE(nn.Module):
    def __init__(self, num_node_features, hidden_dim, last_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, last_dim)
        self.bn2 = nn.LayerNorm(last_dim)

    def forward(self, data):
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
    
    def __init__(self, embedding_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Linear(embedding_dim, 1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        attention_scores = self.attention_fc(embeddings) 
        attention_weights = F.softmax(attention_scores, dim=0) 

        weighted_embeddings = embeddings * attention_weights
        aggregated_embeddings = torch.sum(weighted_embeddings, dim=0) 
        return aggregated_embeddings, attention_weights.squeeze(-1).transpose(0, 1)



def LinearBlock(input_dim, output_dim, p_drop):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


class SpaCLR(nn.Module):
    def __init__(self, num_node_features, hidden_dim, last_dim, image_dims, p_drop, n_pos, backbone='densenet', projection_dims=[64, 64]):
        super(SpaCLR, self).__init__()
        self.gene_encoder = GraphSAGE(num_node_features, hidden_dim, last_dim)
        self.set2set = Set2Set(last_dim, processing_steps=3)
        self.set2set_linear = nn.Linear(2 * last_dim, last_dim)
        self.attention_layer = AttentionLayer(last_dim)
        self.mse_loss = nn.MSELoss()

        if backbone == 'densenet':
            self.image_encoder = densenet121(pretrained=True)
            n_features = self.image_encoder.classifier.in_features
            self.image_encoder.classifier = nn.Identity()
        elif backbone == 'resnet':
            self.image_encoder = resnet50(pretrained=True)
            n_features = self.image_encoder.fc.in_features
            self.image_encoder.fc = nn.Identity()

        self.x_embedding = nn.Embedding(n_pos, n_features)
        self.y_embedding = nn.Embedding(n_pos, n_features)

        image_dims[0] = n_features
        image_dims.append(projection_dims[0])
        self.image_linear = nn.Sequential(OrderedDict([
            (f'image_block{i+1}', LinearBlock(image_dims[i], image_dims[i+1], p_drop)) for i, _ in enumerate(image_dims[:-1])
        ]))

        self.projector = nn.Sequential(
            nn.Linear(projection_dims[0], projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1]),
        )

    def forward_image(self, xi, spatial):
        xi = self.image_encoder(xi)
        xi = self.image_linear(xi)
        hi = self.projector(xi)
        return xi, hi

    def forward_gene(self, xg_list):
        all_embeddings = []
        all_batches = []

        # Process all graphs in one go
        for xg_batch in xg_list:
            embeddings = self.gene_encoder(xg_batch)
            all_embeddings.append(embeddings)
            all_batches.append(xg_batch.batch)

        # Apply set2set to all embeddings
        graph_embeddings_list = []
        for embeddings, batch in zip(all_embeddings, all_batches):
            graph_embeddings = self.set2set(embeddings, batch)
            graph_embeddings = self.set2set_linear(graph_embeddings)
            graph_embeddings_list.append(graph_embeddings)

        all_embeddings = torch.stack(graph_embeddings_list)
        all_embeddings = all_embeddings.transpose(0, 1)  # [B, N_pathways, F] -> [N_pathways, B, F]
        aggregated_embeddings, attention_weights = self.attention_layer(all_embeddings)
        hg = self.projector(aggregated_embeddings)
        return aggregated_embeddings, hg, attention_weights

    def forward(self, xg_list, xi, spatial):
        xg, hg, pathway_attn_weights = self.forward_gene(xg_list)
        xi, hi = self.forward_image(xi, spatial)
        return xg, xi, hg, hi, pathway_attn_weights



class TrainerSpaCLR:
    def __init__(self, args, n_clusters, network, optimizer, log_dir, device='cuda'):
        self.n_clusters = n_clusters
        self.network = network
        self.optimizer = optimizer
        self.train_writer = SummaryWriter(log_dir+'_train')
        self.valid_writer = SummaryWriter(log_dir+'_valid')
        self.device = device

        # Track best ARI and predictions
        self.best_ari = -1.0
        self.best_pred_labels = None
        self.best_epoch = 0

        self.args = args
        if args.dataset == "SpatialLIBD":
            adata = load_ST_file(os.path.join(args.path, args.name))
            df_meta = pd.read_csv(os.path.join(args.path, args.name, 'metadata.tsv'), sep='\t')
            label = pd.Categorical(df_meta['layer_guess']).codes
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
            label = pd.Categorical(df_meta['annot_type']).codes
            # label = pd.Categorical(df_meta['ground_truth']).codes
            n_clusters = label.max() + 1
            adata = adata[label != -1]
            self.sample_id = adata.obs.index.tolist()
            self.adj_2d = calculate_adj_matrix(x=adata.obs["array_row"].tolist(), y=adata.obs["array_col"].tolist(),
                                      histology=False)
            
        elif args.dataset=="MBA":
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
        self.w_recon = args.w_recon
        self.temperature = args.temperature
        self.csv_path = os.path.join('results',
                                     f'{args.name}_last_dim_{args.last_dim}_hidden_dim_{args.hidden_dim}_prob_edge_{args.prob_edge_perturb}_pct_edge_perturb_{args.pct_edge_perturb}_p_drop_{args.p_drop}_lr_{args.lr}_temp_{args.temperature}_results.csv')
        csv_directory = os.path.dirname(self.csv_path)
        os.makedirs(csv_directory, exist_ok=True)
        
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'ARI', 'Train Loss', 'Validation Loss', 'Learning Rate'])
    
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
        with tqdm(total=len(trainloader)) as t:
            self.network.train()
            train_loss = 0
            train_cnt = 0

            for i, batch in enumerate(trainloader):
                t.set_description(f'Epoch {epoch} train')

                self.optimizer.zero_grad()
                xg, xg_u, xg_v, xi_u, xi_v, spatial, y, _ = batch

                xg = [item.to(self.device) for item in xg]
                xg_u = [item.to(self.device) for item in xg_u]
                xg_v = [item.to(self.device) for item in xg_v]
                xi_u = xi_u.to(self.device)
                xi_v = xi_v.to(self.device)
                spatial = spatial.to(self.device)

                _, hg, _ = self.network.forward_gene(xg)
                _, hg_u, _ = self.network.forward_gene(xg_u)
                _, hg_v, _ = self.network.forward_gene(xg_v)

                _, hi_u = self.network.forward_image(xi_u, spatial)
                _, hi_v = self.network.forward_image(xi_v, spatial)
                criterion = NT_Xent(hg.shape[0], temperature=self.temperature)

                g2g_loss = criterion(hg_u, hg_v) * self.w_g2g
                i2i_loss = criterion(hi_u, hi_v) * self.w_i2i
                g2i_loss = criterion(hg, hi_u)
                loss = g2i_loss + g2g_loss + i2i_loss

                loss.backward()
                self.optimizer.step()

                train_cnt += 1
                train_loss += loss.item()

                t.set_postfix(loss=f'{(train_loss/train_cnt):.3f}',
                            g2i_loss=f'{g2i_loss.item():.3f}',
                            g2g_loss=f'{g2g_loss.item():.3f}',
                            i2i_loss=f'{i2i_loss.item():.3f}',)
                t.update(1)
            avg_train_loss = train_loss / train_cnt
            self.train_writer.add_scalar('loss', (train_loss/train_cnt), epoch)
            self.train_writer.flush()

            return avg_train_loss
            
    def valid(self, validloader, epoch=0):
        Xg = []
        Xi = []
        Y = []
        all_attention_weights = []

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

                    xg, xi, hg, hi, attention_weights = self.network(xg, xi, spatial)
                    criterion = NT_Xent(xg.shape[0], temperature=self.temperature)
                    loss = criterion(hg, hi)

                    valid_cnt += 1
                    valid_loss += loss.item()

                    Xg.append(hg.detach().cpu().numpy())
                    Xi.append(hi.detach().cpu().numpy())
                    Y.append(y)
                    all_attention_weights.append(attention_weights.detach().cpu().numpy())
                    t.set_postfix(loss=f'{(valid_loss / valid_cnt):.3f}')
                    t.update(1)

                Xg = np.vstack(Xg)
                Xi = np.vstack(Xi)
                Y = np.concatenate(Y, 0)
                all_attention_weights = np.concatenate(all_attention_weights, axis=0)

        attention_weights_path = os.path.join('attention', f'attention_weights_epoch{epoch}_sage_.npy')
        os.makedirs(os.path.dirname(attention_weights_path), exist_ok=True)
        np.save(attention_weights_path, all_attention_weights)

        return Xg, Xi, Y, valid_loss / valid_cnt
    
    def fit(self, trainloader, validloader, epochs, dataset, name, path, checkpoint_path=None):
        self.network = self.network.to(self.device)
        start_epoch = 0

        if checkpoint_path is not None:
            self.load_model(checkpoint_path)
            start_epoch = int(checkpoint_path.split('_epoch')[1].split('_')[0])

        for epoch in range(epochs):
            avg_train_loss = self.train(trainloader, epoch + 1)
            Xg, Xi, Y, val_loss = self.valid(validloader, epoch + 1)
            z = Xg + Xi
            print(f"[DEBUG] Before sum: Xg.shape={Xg.shape}, Xi.shape={Xi.shape}")
            ari, pred_labels = get_predicted_results(dataset, name, path, z)

            # Track best ARI and predictions
            if ari > self.best_ari:
                self.best_ari = ari
                self.best_pred_labels = pred_labels
                self.best_epoch = epoch + 1
                print(f"[INFO] New best ARI: {self.best_ari:.4f} at epoch {self.best_epoch}")

            lr = self.optimizer.param_groups[0]['lr']
            with open(self.csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch + 1, ari, avg_train_loss, val_loss, lr])

            embedding_dir = 'embeddings'
            os.makedirs(embedding_dir, exist_ok=True)
            np.save(os.path.join(embedding_dir, f'xg_epoch{epoch+1}_SAGE.npy'), Xg)
            np.save(os.path.join(embedding_dir, f'xi_epoch{epoch+1}_SAGE.npy'), Xi)

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
        return xg + xi

    def save_model(self, ckpt_path):
        torch.save(self.network.state_dict(), ckpt_path)

    def load_model(self, ckpt_path):
        self.network.load_state_dict(torch.load(ckpt_path))