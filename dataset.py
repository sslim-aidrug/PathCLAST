import os
from copy import deepcopy

import numpy as np
import pandas as pd
import cv2
import networkx as nx
from scipy.sparse import issparse
import torch
from torch import nn
from torch.utils import data
from torch_geometric.data import Data
from torchvision import transforms
from torchtoolbox.transform import Cutout

from utils import load_ST_file, build_her2st_data


class PathwayProcessor:
    """
    Build and update pathway graphs with per-gene expression
    """

    def __init__(self, csv_file):
        """

        pathway CSV containing columns:
        ['PathwayID','NodeInfo','EdgeInfo']
        """
        self.prebuilt_graphs = {}
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['EdgeInfo'].notnull()]
        
    def create_static_pathway_graphs(self):
        """
        Parse CSV rows and create one static NetworkX graph per pathway.
        """
        graphs = {}

        for pid, grp in self.df.groupby('PathwayID'):
            G = nx.Graph()

            # nodes
            node_info = str(grp['NodeInfo'].iloc[0])
            nodes = [n.strip() for n in node_info.split(';') if n.strip()]
            G.add_nodes_from(nodes)

            # edges
            edge_info = str(grp['EdgeInfo'].iloc[0])
            edges = []
            for e in edge_info.split(';'):
                e = e.strip()
                if not e:
                    continue
                e = e.strip('()')
                parts = [p.strip() for p in e.split(',')]
                if len(parts) == 2 and parts[0] and parts[1]:
                    edges.append((parts[0], parts[1]))
            if edges:
                G.add_edges_from(edges)

            graphs[pid] = G

        self.prebuilt_graphs = graphs
        print(f"Total pathways: {len(self.prebuilt_graphs)}")

    def update_graph_with_expression(self, sPathID, expression_data_matrix, symbol_to_index):
        """
        Attach per-gene expression to nodes in a copied pathway graph
        return: nx.Graph with node['expression_data'] = np.ndarray(shape=(1,)
        """
        if sPathID not in self.prebuilt_graphs:
            return None

        G = deepcopy(self.prebuilt_graphs[sPathID])

        pathway_genes = [node for node in G.nodes() if node in symbol_to_index]
        if len(pathway_genes) == 0:
            return None  # or handle the empty case appropriately

        pathway_expression_data = expression_data_matrix[:, [symbol_to_index[gene] for gene in pathway_genes]]

        for node in G.nodes():
            gene_symbol = node  
            if gene_symbol in symbol_to_index:
                idx = pathway_genes.index(gene_symbol)
                expression_data = pathway_expression_data[:, idx]
                G.nodes[node]['expression_data'] = expression_data
            else:
                G.nodes[node]['expression_data'] = np.zeros(expression_data_matrix.shape[0], dtype=np.float32)

        return G
            

class Dataset(data.Dataset):
    """
    Bi-modal ST dataset that builds pathway graphs per spot
    """

    def __init__(self,
                 dataset, path, name, csv_file,
                 prob_node_drop=0.5, pct_node_drop=0.15,
                 prob_edge_perturb=0.5, pct_edge_perturb=0.15,
                 img_size=112, train=True):
        super().__init__()

        self.dataset = dataset
        self.train = train

        # prebuild static pathway graphs once
        self.processor = PathwayProcessor(csv_file)
        self.processor.create_static_pathway_graphs()

        
        if dataset == "DLPFC":
            adata = load_ST_file(os.path.join(path, name))
            adata.X = adata.X.A
            df_meta = pd.read_csv(os.path.join(path, name, 'metadata.tsv'), sep='\t')
            self.label = pd.Categorical(df_meta['ground_truth']).codes
            full_image = cv2.imread(os.path.join(path, name, f'{name}_full_image.tif'))
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            patches = []
            for x, y in adata.obsm['spatial']:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                patches.append(full_image[y-img_size:y+img_size, x-img_size:x+img_size])
            patches = np.array(patches)
            self.image = patches

        elif dataset == "Her2st":
            adata, patches = build_her2st_data(path, name, img_size)
            self.label = adata.obs['label']
            self.image = patches

        elif dataset == "IDC":
            adata = load_ST_file(os.path.join(path, name))
            adata.X = adata.X.A
            self.label = np.zeros(adata.shape[0], dtype=int)

            full_image = cv2.imread(os.path.join(path, name, f'{name}.tif'))
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            patches = []
            for x, y in adata.obsm['spatial']:
                patches.append(full_image[y - img_size:y + img_size, x - img_size:x + img_size])
            patches = np.array(patches)
            self.image = patches


        self.n_clusters = self.label.max() + 1
        self.spatial = adata.obsm['spatial']
        self.n_pos = self.spatial.max() + 1
        
        self.gene = adata
        self.gene = self.gene[self.label != -1]      
        self.image = self.image[self.label != -1]
        self.label = self.label[self.label != -1]

        
        expression_idx = list(self.gene.var_names)
        self.symbol_to_index = {symbol: idx for idx, symbol in enumerate(expression_idx)}

        # --- image augmentation ---
        self.img_train_transform = transforms.Compose([
            Cutout(0.5),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.img_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # --- graph augmentation ---
        self.graph_train_transform = GraphGeneTransforms(
            prob_node_drop=prob_node_drop, pct_node_drop=pct_node_drop,
            prob_edge_perturb=prob_edge_perturb, pct_edge_perturb=pct_edge_perturb
        )

    def update_graphs_with_expression(self, xg):
        """
        Build per-pathway graphs with node features.

        :param xg: AnnData (1 x n_genes)
        :return: dict {pathway_id: nx.Graph(with node['expression_data']: np.ndarray (1,)
        """
        updated_graphs = {}

        #
        X = xg.X
        if issparse(X):
            gene_matrix = X.toarray()
        elif isinstance(X, np.ndarray):
            gene_matrix = X
        else:
            
            gene_matrix = getattr(X, "values", np.asarray(X))

        gene_matrix = gene_matrix.astype(np.float32, copy=False)

        # update each pathway graph
        for sPathID, _ in self.processor.prebuilt_graphs.items():
            g_upd = self.processor.update_graph_with_expression(
                sPathID, gene_matrix, self.symbol_to_index
            )
            if g_upd:
                updated_graphs[sPathID] = g_upd

        return updated_graphs

    def convert_graphs_to_tensor(self, graphs):
        """
        Convert pathway graphs into torch_geometric Data objects
        """
        pyg_list = []

        for _, G in graphs.items():
            node_indices = {node: i for i, node in enumerate(G.nodes())}

            # edges -> [2, E]
            if G.number_of_edges() > 0:
                edges = [[node_indices[u], node_indices[v]] for u, v in G.edges()]
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # node features (expression_data) -> [N, 1]
            feats = []
            for node in G.nodes():
                expr = G.nodes[node].get('expression_data', np.zeros(1)).reshape(-1, 1)
                feats.append(expr)
            node_features = torch.tensor(np.asarray(feats), dtype=torch.float).squeeze()
            if node_features.dim() == 1:
                node_features = node_features.unsqueeze(1)

            pyg_list.append(Data(x=node_features, edge_index=edge_index))

        return pyg_list

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        """
        Return:
        graphs, graph augs, image augs, spatial, label, idx.
        """
        spatial = torch.from_numpy(self.spatial[idx])
        y = int(self.label[idx])
        xg_row = self.gene[idx] 

        # per-spot pathway graphs
        updated_graphs = self.update_graphs_with_expression(xg_row)
        xg = self.convert_graphs_to_tensor(updated_graphs)

        if self.train:
            # graph augmentations
            xg_u = [self.graph_train_transform(deepcopy(g)) for g in xg]
            xg_v = [self.graph_train_transform(deepcopy(g)) for g in xg]

            # image augmentations
            xi_u = self.img_train_transform(self.image[idx])
            xi_v = self.img_train_transform(self.image[idx])

            return xg, xg_u, xg_v, xi_u, xi_v, spatial, y, idx

        xi = self.img_test_transform(self.image[idx])
        return xg, xi, spatial, y, idx

class GraphGeneTransforms(nn.Module):
    """
    Graph-level data augmentations: node drop and edge perturbation
    """

    def __init__(self, prob_node_drop=0.5, pct_node_drop=0.15,
                 prob_edge_perturb=0.5, pct_edge_perturb=0.15):
        super(GraphGeneTransforms, self).__init__()
        self.prob_node_drop = prob_node_drop
        self.pct_node_drop = pct_node_drop
        self.prob_edge_perturb = prob_edge_perturb
        self.pct_edge_perturb = pct_edge_perturb

    def forward(self, data):
        """
        Apply node dropping and edge perturbation to input graph
        """
        xg, edge_index = data.x, data.edge_index
        num_nodes = xg.size(0)

        # node dropping
        if torch.rand(1) < self.prob_node_drop:
            drop_num = int(num_nodes * self.pct_node_drop)
            keep_indices = torch.randperm(num_nodes)[drop_num:]  # directly choose kept nodes
            xg = xg[keep_indices]

            node_map = {old: new for new, old in enumerate(keep_indices.tolist())}
            new_edges = [
                (node_map[s.item()], node_map[d.item()])
                for s, d in edge_index.t()
                if s.item() in node_map and d.item() in node_map
            ]
            edge_index = (torch.tensor(new_edges, dtype=torch.long).t().contiguous()
                          if new_edges else torch.empty((2, 0), dtype=torch.long))
        else:
            keep_indices = torch.arange(num_nodes)

        # edge perturbation
        if edge_index.size(1) > 0 and torch.rand(1) < self.prob_edge_perturb:
            edge_num = edge_index.size(1)
            perturb_num = int(edge_num * self.pct_edge_perturb)

            # directly sample kept edges
            kept_edges = torch.randperm(edge_num)[perturb_num:]
            edge_index = edge_index[:, kept_edges]

            # add random new edges
            new_edges = torch.randint(0, len(keep_indices), (2, perturb_num))
            edge_index = torch.cat([edge_index, new_edges], dim=1)

        data.x, data.edge_index = xg, edge_index
        return data

