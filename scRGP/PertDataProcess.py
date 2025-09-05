import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import scanpy as sc
from tqdm import tqdm
from scipy.stats import rankdata
from utils import get_genes_from_perts, DataSplitter

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0


class PertData:
    def __init__(self, data_path):
        self.data_path = data_path

    def load(self, data_name=None, test_file=None, mode=''):
        """Load the h5ad data and build/read the cell graph dataset"""
        if not os.path.exists(self.data_path):
            raise ValueError("h5ad file path does not exist")

        data_path = os.path.join(self.data_path, data_name)
        adata_file = test_file or 'perturb_processed.h5ad'
        self.adata = sc.read_h5ad(os.path.join(data_path, adata_file))
        self.dataset_name = os.path.basename(data_path)
        self.dataset_path = data_path

        if mode == 'debug':
            np.random.seed(1)
            self.adata = self.adata[np.random.choice(len(self.adata), 200, replace=False)]

        self.pert_names = get_genes_from_perts(self.adata.obs['condition'].unique())
        self.gene_names = self.adata.var.gene_name.tolist()

        # Check the local cache
        dataset_fname = os.path.join(data_path, 'data_pyg', 'cell_graphs.pkl')
        os.makedirs(os.path.dirname(dataset_fname), exist_ok=True)

        if os.path.isfile(dataset_fname):
            print("Loading cached pyg dataset...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))
        else:
            print("Creating pyg dataset...")
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            self.dataset_processed = {
                p: self.create_cell_graph_dataset(self.adata, p)
                for p in tqdm(self.adata.obs['condition'].unique())
            }
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))
            print("Saved dataset to", dataset_fname)

    def prepare_split(self, split='single', seed=1, train_gene_set_size=0.8, test_gene_set_size=0.1, split_dict_path=None):
        """train/val/test split"""
        valid_splits = ['combo_seen0', 'combo_seen1', 'combo_seen2', 'single',
                        'no_test', 'only_test', 'only_train', 'load_split']
        if split not in valid_splits:
            raise ValueError(f"Invalid split, must be one of {valid_splits}")

        self.split, self.seed = split, seed
        split_folder = os.path.join(self.dataset_path, 'splits')
        os.makedirs(split_folder, exist_ok=True)
        split_file = f"{self.dataset_name}_{split}_{seed}_{train_gene_set_size}.pkl"
        split_path = os.path.join(split_folder, split_file)

        if split == 'load_split':
            if not split_dict_path:
                raise ValueError("split_dict_path required for load_split")
            self.set2conditions = pickle.load(open(split_dict_path, 'rb'))
            return

        if os.path.exists(split_path):
            print("Loading cached split...")
            self.set2conditions = pickle.load(open(split_path, "rb"))
            return

        print("Creating new split...")
        if split.startswith('combo'):
            DS = DataSplitter(self.adata, split_type='combo', seen=int(split[-1]))
            adata = DS.split_data(test_size=test_gene_set_size, seed=seed)
        elif split in ['single', 'no_test']:
            DS = DataSplitter(self.adata, split_type=split)
            adata = DS.split_data(test_size=test_gene_set_size, seed=seed) if split == 'single' else DS.split_data(seed=seed)
        elif split in ['only_test', 'only_train']:
            adata = self.adata.copy()
            adata.obs['split'] = split.split('_')[-1]
        else:
            raise ValueError(f"Unsupported split: {split}")

        set2conditions = dict(adata.obs.groupby('split')['condition'].unique())
        self.set2conditions = {k: v.tolist() for k, v in set2conditions.items()}
        pickle.dump(self.set2conditions, open(split_path, "wb"))
        print("Saved split to", split_path)

    def get_dataloader(self, batch_size, test_batch_size=None):
        """return DataLoader"""
        test_batch_size = test_batch_size or batch_size
        cell_graphs = {}

        def collect_graphs(split_name):
            return [g for p in self.set2conditions.get(split_name, []) if p != 'ctrl'
                    for g in self.dataset_processed[p]]

        if self.split in ['only_test', 'only_train']:
            split_name = self.split.split('_')[-1]
            loader = DataLoader(collect_graphs(split_name),
                                batch_size=batch_size, shuffle=False)
            self.dataloader = {f"{split_name}_loader": loader}
            return self.dataloader

        splits = ['train', 'val'] if self.split == 'no_test' else ['train', 'val', 'test']
        for s in splits:
            cell_graphs[s] = collect_graphs(s)

        self.dataloader = {
            'train_loader': DataLoader(cell_graphs['train'], batch_size=batch_size, shuffle=True, drop_last=True),
            'val_loader': DataLoader(cell_graphs['val'], batch_size=batch_size, shuffle=True)
        }
        if 'test' in cell_graphs:
            self.dataloader['test_loader'] = DataLoader(cell_graphs['test'], batch_size=test_batch_size, shuffle=False)
        return self.dataloader

    def create_cell_graph_dataset(self, adata, pert_category):
        """Build a cell graph dataset for a certain perturbation"""
        adata_ = adata[adata.obs['condition'] == pert_category]
        de_genes = adata_.uns.get('rank_genes_groups_cov_all', None)
        de = de_genes is not None

        if pert_category != 'ctrl':
            try:
                pert_idx = [np.where(p == np.array(self.gene_names))[0][0]
                            for p in pert_category.split('+') if p != 'ctrl']
            except:
                raise ValueError(f"{pert_category} not in gene list")

            pert_de_category = adata_.obs.get('condition_name', [pert_category])[0]
            de_idx = {k: np.where(adata_.var_names.isin(np.array(de_genes[pert_de_category][:k])))[0] if de else [-1] * k
                      for k in [20, 50, 100, 200]}

            Xs, ys = [], []
            for cell_z in adata_.X:
                for c in self.ctrl_adata[np.random.randint(0, len(self.ctrl_adata), 1), :].X:
                    Xs.append(c)
                    ys.append(cell_z)

        else:
            pert_idx = None
            de_idx = {k: [-1] * k for k in [20, 50, 100, 200]}
            Xs, ys = adata_.X, adata_.X

        return [self.create_cell_graph(x.toarray(), y.toarray(), de_idx, pert_category, pert_idx)
                for x, y in zip(Xs, ys)]

    @staticmethod
    def create_cell_graph(X, y, de_idx, pert, pert_idx=None):
        """Build a graph of a single cell"""
        X, y = rankdata(X, method='dense'), rankdata(y, method='dense')
        return Data(
            x=torch.Tensor(X).T,
            y=torch.Tensor(y),
            pert=pert,
            pert_idx=pert_idx or [-1],
            **{f"de_idx_{k}": v for k, v in de_idx.items()}
        )
