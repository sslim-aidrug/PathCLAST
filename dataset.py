import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from torch_geometric.data import Data
from scipy.sparse import issparse
from torchvision import transforms
from torchtoolbox.transform import Cutout
import cv2
import scanpy as sc

from utils import load_ST_file, build_her2st_data


class PathwayProcessor:
    """
    Optimized pathway processor using index-based approach.
    Pre-computes pathway gene indices and edge structures for fast runtime access.
    """
    def __init__(self, csv_file, all_genes_list):
        df = pd.read_csv(csv_file)
        df = df[df['EdgeInfo'].notnull()]

        # Create gene name to index mapping
        self.all_genes_map = {gene: i for i, gene in enumerate(all_genes_list)}

        self.pathway_ids = []
        self.pathway_gene_indices = []  # Each pathway as list of gene indices
        self.pathway_edge_indices = []  # Pre-computed edge_index tensors

        print("Pre-computing pathway structures...")
        for _, row in df.iterrows():
            pathway_id = row['PathwayID']
            self.pathway_ids.append(pathway_id)

            # Parse nodes and convert to indices
            nodes = [node.strip() for node in row['NodeInfo'].split(';')]
            gene_indices = [self.all_genes_map[gene] for gene in nodes if gene in self.all_genes_map]
            self.pathway_gene_indices.append(gene_indices)

            # Parse edges and build edge_index
            edge_info = row['EdgeInfo']
            if pd.notna(edge_info) and len(gene_indices) > 0:
                # Create node to local index mapping for this pathway
                # IMPORTANT: Only assign indices to genes that exist in all_genes_map
                valid_nodes = [gene for gene in nodes if gene in self.all_genes_map]
                node_to_local_idx = {gene: idx for idx, gene in enumerate(valid_nodes)}

                edges = []
                for edge in edge_info.split(';'):
                    if edge.strip():
                        try:
                            # Parse edge format: "(gene1, gene2)"
                            edge_clean = edge.strip('() ')
                            src, dst = edge_clean.split(',')
                            src, dst = src.strip(), dst.strip()

                            if src in node_to_local_idx and dst in node_to_local_idx:
                                edges.append([node_to_local_idx[src], node_to_local_idx[dst]])
                        except:
                            continue

                if edges:
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                else:
                    # No valid edges, create empty edge_index
                    edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                # No edges or no valid genes
                edge_index = torch.empty((2, 0), dtype=torch.long)

            self.pathway_edge_indices.append(edge_index)

        print(f"Pre-computed {len(self.pathway_ids)} pathways")

    def get_num_pathways(self):
        return len(self.pathway_ids)


class Dataset(data.Dataset):
    def __init__(self, dataset, path, name, csv_file,
                 prob_node_drop=0.5, pct_node_drop=0.15,
                 prob_edge_perturb=0.5, pct_edge_perturb=0.15, img_size=112, train=True):

        self.dataset = dataset
        self.train = train

        # Load dataset
        if dataset == "SpatialLIBD":
            adata = load_ST_file(os.path.join(path, name))
            adata.X = adata.X.A
            df_meta = pd.read_csv(os.path.join(path, name, 'metadata.tsv'), sep='\t')
            self.label = pd.Categorical(df_meta['layer_guess']).codes
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

        elif dataset == "MBA":
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

        # Store gene expression as numpy array for fast indexing
        if issparse(adata.X):
            self.gene_expression = adata.X.toarray().astype(np.float32)
        else:
            self.gene_expression = np.asarray(adata.X, dtype=np.float32)

        self.n_clusters = self.label.max() + 1
        self.spatial = adata.obsm['spatial']
        self.n_pos = self.spatial.max() + 1

        # Initialize pathway processor with gene list
        self.all_genes_list = list(adata.var_names)
        self.processor = PathwayProcessor(csv_file, self.all_genes_list)

        # Filter out invalid samples
        self.gene_expression = self.gene_expression[self.label != -1]
        self.image = self.image[self.label != -1]
        self.spatial = self.spatial[self.label != -1]
        self.label = self.label[self.label != -1]

        # Image transforms
        self.img_train_transform = transforms.Compose([
            Cutout(0.5),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Graph augmentation parameters
        self.prob_node_drop = prob_node_drop
        self.pct_node_drop = pct_node_drop
        self.prob_edge_perturb = prob_edge_perturb
        self.pct_edge_perturb = pct_edge_perturb

        print(f"Dataset initialized: {len(self.label)} samples, {len(self.all_genes_list)} genes, {self.processor.get_num_pathways()} pathways")

    def _create_pathway_graphs(self, spot_expression):
        """
        Create PyG Data objects for all pathways using pre-computed structures.
        This is called per sample but uses optimized numpy indexing.
        """
        pyg_data_objects = []

        for i in range(self.processor.get_num_pathways()):
            gene_indices = self.processor.pathway_gene_indices[i]
            edge_index = self.processor.pathway_edge_indices[i]

            if len(gene_indices) == 0:
                continue

            # Fast numpy indexing to extract pathway expression
            pathway_expression = spot_expression[gene_indices]

            # Create node features (each node is one gene with its expression value)
            node_features = torch.from_numpy(pathway_expression).float().unsqueeze(1)

            # Create PyG Data object
            pyg_data_objects.append(Data(x=node_features, edge_index=edge_index))

        return pyg_data_objects

    def _augment_graph(self, data):
        """Apply graph augmentation (node drop + edge perturbation)"""
        xg = data.x
        num_nodes = xg.size(0)
        edge_index = data.edge_index

        # Node drop
        if torch.rand(1) < self.prob_node_drop and num_nodes > 1:
            drop_num = max(1, int(num_nodes * self.pct_node_drop))
            keep_num = num_nodes - drop_num
            keep_indices = torch.randperm(num_nodes)[:keep_num]
            keep_indices_sorted, _ = torch.sort(keep_indices)

            xg = xg[keep_indices_sorted]

            # Update edge_index
            node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(keep_indices_sorted)}
            new_edges = []
            for src, dst in edge_index.t():
                if src.item() in node_map and dst.item() in node_map:
                    new_edges.append([node_map[src.item()], node_map[dst.item()]])

            if new_edges:
                edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            num_nodes = keep_num

        # Edge perturbation
        if edge_index.size(1) > 0 and torch.rand(1) < self.prob_edge_perturb:
            edge_num = edge_index.size(1)
            edge_perturb_num = max(1, int(edge_num * self.pct_edge_perturb))

            # Remove some edges
            keep_num = edge_num - edge_perturb_num
            keep_edge_indices = torch.randperm(edge_num)[:keep_num]
            edge_index = edge_index[:, keep_edge_indices]

            # Add random edges
            if num_nodes > 1:
                new_edges = torch.randint(0, num_nodes, (2, edge_perturb_num))
                edge_index = torch.cat([edge_index, new_edges], dim=1)

        data.x = xg
        data.edge_index = edge_index
        return data

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        spatial = torch.from_numpy(self.spatial[idx])
        y = self.label[idx]

        # Fast indexing to get spot expression
        spot_expression = self.gene_expression[idx]

        # Create pathway graphs using optimized method
        xg = self._create_pathway_graphs(spot_expression)

        if self.train:
            # Apply augmentations
            xg_u = [self._augment_graph(graph.clone()) for graph in xg]
            xg_v = [self._augment_graph(graph.clone()) for graph in xg]
            xi_u = self.img_train_transform(self.image[idx])
            xi_v = self.img_train_transform(self.image[idx])
            return xg, xg_u, xg_v, xi_u, xi_v, spatial, y, idx
        else:
            xi = self.img_test_transform(self.image[idx])
            return xg, xi, spatial, y, idx
