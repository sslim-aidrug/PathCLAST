import os
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from torch_geometric.data import Batch as GeoBatch

from dataset import Dataset
from model import PathCLAST, TrainerPathCLAST
from utils import get_predicted_results


def seed_torch(seed: int):
    """Seed Python/NumPy/PyTorch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def custom_collate_fn(batch):
    """
    Collate function for per-sample pathway lists and image views

    """
    if len(batch[0]) == 8:
        xg_list, xg_u_list, xg_v_list, xi_u, xi_v, spatial, y, idx_returned = zip(*batch)

        xg   = [GeoBatch.from_data_list(sublist) for sublist in xg_list]
        xg_u = [GeoBatch.from_data_list(sublist) for sublist in xg_u_list]
        xg_v = [GeoBatch.from_data_list(sublist) for sublist in xg_v_list]

        xi_u = torch.stack(xi_u, dim=0)
        xi_v = torch.stack(xi_v, dim=0)
        spatial = torch.stack(spatial, dim=0)
        y = torch.tensor(y, dtype=torch.long)
        idx_returned = torch.tensor(idx_returned, dtype=torch.long)
        return xg, xg_u, xg_v, xi_u, xi_v, spatial, y, idx_returned

    # eval
    xg_list, xi, spatial, y, idx_returned = zip(*batch)
    xg = [GeoBatch.from_data_list(sublist) for sublist in xg_list]
    xi = torch.stack(xi, dim=0)
    spatial = torch.stack(spatial, dim=0)
    y = torch.tensor(y, dtype=torch.long)
    idx_returned = torch.tensor(idx_returned, dtype=torch.long)
    return xg, xi, spatial, y, idx_returned


def train(args, name: str):

    seed_torch(1)

    path = args.path
    last_dim = args.last_dim
    hidden_dim = args.hidden_dim
    image_dims = [last_dim]
    lr = args.lr
    p_drop = args.p_drop
    batch_size = args.batch_size
    dataset = args.dataset
    epochs = args.epochs
    img_size = args.img_size
    device = args.device
    log_name = args.log_name
    num_workers = args.num_workers
    prob_edge_perturb = args.prob_edge_perturb
    pct_edge_perturb = args.pct_edge_perturb
    prob_node_drop = args.prob_node_drop
    pct_node_drop = args.pct_node_drop
    weight_decay = args.weight_decay

    csv_path = "/path/hsa_pathway.csv"
    trainset = Dataset(
        dataset=dataset, path=path, name=name, csv_file=csv_path,
        prob_edge_perturb=prob_edge_perturb, pct_edge_perturb=pct_edge_perturb,
        prob_node_drop=prob_node_drop, pct_node_drop=pct_node_drop,
        img_size=img_size, train=True
    )
    testset = Dataset(
        dataset=dataset, path=path, name=name, csv_file=csv_path,
        prob_edge_perturb=prob_edge_perturb, pct_edge_perturb=pct_edge_perturb,
        prob_node_drop=prob_node_drop, pct_node_drop=pct_node_drop,
        img_size=img_size, train=False
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )

    # model
    network = PathCLAST(
        num_node_features=1, last_dim=last_dim, hidden_dim=hidden_dim,
        image_dims=image_dims, p_drop=p_drop, n_pos=trainset.n_pos, backbone="densenet",
        projection_dims=[last_dim, last_dim]
    )

    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)

    # log path
    save_name = f"{name}_{args.w_g2g}_{args.w_i2i}"
    log_dir = os.path.join("log", log_name, save_name)

    # train / validate
    trainer = TrainerPathCLAST(args, trainset.n_clusters, network, optimizer, log_dir, device=device)
    trainer.fit(trainloader, epochs)

    xg, xi, _, _ = trainer.valid(testloader)
    z = xg + xi
    ari, pred_label = get_predicted_results(dataset, name, path, z)
    print("Ari value :", ari)

    # save predictions
    os.makedirs("output", exist_ok=True)
    pd.DataFrame({"cluster_labels": pred_label}).to_csv(f"output/{name}_pred.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset / paths
    parser.add_argument("--dataset", type=str, default="DLPFC")
    parser.add_argument("--path", type=str, default="path/")
    parser.add_argument("--name", type=str, default="151507")
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--num_workers", type=int, default=8)

    # model
    parser.add_argument("--last_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--p_drop", type=float, default=0.3)

    # optimizer
    parser.add_argument("--lr", type=float, default=0.007)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # loss weights
    parser.add_argument("--w_g2g", type=float, default=0.3)
    parser.add_argument("--w_i2i", type=float, default=0.2)

    # graph augmentation
    parser.add_argument("--prob_edge_perturb", type=float, default=0.3)
    parser.add_argument("--pct_edge_perturb", type=float, default=0.1)
    parser.add_argument("--prob_node_drop", type=float, default=0.3)
    parser.add_argument("--pct_node_drop", type=float, default=0.5)

    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_name", type=str, default="log_name")

    args = parser.parse_args()
    train(args, args.name)
