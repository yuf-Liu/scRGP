import scanpy as sc
import warnings
import torch
import numpy as np
from scipy.stats import rankdata
from torch_geometric.data import Data

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

def get_genes_from_perts(perts):
    if isinstance(perts, str):
        perts = [perts]

    genes = []
    for p in np.unique(perts):
        genes.extend(p.split('+'))

    genes = [g for g in genes if g != 'ctrl']
    return sorted(set(genes))


def parse_any_pert(pert):
    if 'ctrl' in pert and pert != 'ctrl':
        a, b = pert.split('+')
        return b if a == 'ctrl' else a
    elif 'ctrl' not in pert:
        return pert.split('+')
    return None


def loss_fct(pred, y, perts, ctrl=None, direction_lambda=1e-3, dict_filter=None):
    gamma = 2
    perts = np.array(perts)
    y = y.reshape_as(pred)

    total_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

    for p in np.unique(perts):
        pert_mask = np.where(perts == p)[0]

        if p != 'ctrl':
            retain_idx = dict_filter[p]
            pred_p, y_p = pred[pert_mask][:, retain_idx], y[pert_mask][:, retain_idx]
            ctrl_ref = ctrl[retain_idx]
        else:
            pred_p, y_p = pred[pert_mask], y[pert_mask]
            ctrl_ref = ctrl

        mse_term = torch.mean((pred_p - y_p) ** (2 + gamma))
        total_loss = total_loss + mse_term

        direction_term = direction_lambda * torch.mean(
            (torch.sign(y_p - ctrl_ref) - torch.sign(pred_p - ctrl_ref)) ** 2
        )
        total_loss = total_loss + direction_term

    return total_loss / len(np.unique(perts))


def create_cell_graph_dataset_for_prediction(pert_gene, ctrl_adata, gene_names,
                                             device, num_samples=300):
    try:
        pert_idx = [
            np.where(np.array(gene_names) == g)[0][0]
            for g in pert_gene.split('+') if g != 'ctrl'
        ]
    except Exception:
        pert_idx = [-1]

    X = ctrl_adata.X.toarray() if hasattr(ctrl_adata.X, "toarray") else ctrl_adata.X

    if num_samples < X.shape[0]:
        sampled_idx = np.random.choice(X.shape[0], num_samples, replace=False)
        X = X[sampled_idx]

    graphs = []
    for expr in X:
        ranked_expr = rankdata(expr, method='dense')
        data = Data(x=torch.tensor(ranked_expr, dtype=torch.float32).unsqueeze(-1),
                    pert_idx=pert_idx,
                    pert=pert_gene)
        graphs.append(data.to(device))

    return graphs


class DataSplitter:
    def __init__(self, adata, split_type='single', seen=0):
        self.adata = adata
        self.split_type = split_type
        self.seen = seen

    def split_data(self, test_size=0.1, test_perts=None, split_name='split',
                   seed=None, val_size=0.1):
        np.random.seed(seed)
        unique_perts = [p for p in self.adata.obs['condition'].unique() if p != 'ctrl']

        if self.split_type == 'no_test':
            train, val = self.get_split_list(unique_perts, test_size=val_size)
            test = []
        else:
            train, test = self.get_split_list(unique_perts, test_size=test_size, test_perts=test_perts)
            train, val = self.get_split_list(train, test_size=val_size)

        mapping = {x: 'train' for x in train}
        mapping.update({x: 'val' for x in val})
        mapping.update({x: 'test' for x in test})
        mapping.update({'ctrl': 'train'})

        self.adata.obs[split_name] = self.adata.obs['condition'].map(mapping)
        return self.adata

    def get_split_list(self, pert_list, test_size=0.1, test_perts=None, hold_outs=True):
        single_perts = [p for p in pert_list if 'ctrl' in p and p != 'ctrl']
        combo_perts = [p for p in pert_list if 'ctrl' not in p]
        unique_genes = self.get_genes_from_perts(pert_list)
        hold_out, test_perts = [], test_perts or []

        # Randomly select some genes as test
        test_pert_genes = np.random.choice(unique_genes, int(len(single_perts) * test_size))

        if self.split_type.startswith('single'):
            test_perts = self.get_perts_from_genes(test_pert_genes, pert_list, 'single')
            hold_out = combo_perts if self.split_type == 'single_only' else \
                       self.get_perts_from_genes(test_pert_genes, pert_list, 'combo')

        elif self.split_type == 'combo':
            test_perts = self._handle_combo_split(combo_perts, test_pert_genes, hold_outs)

        elif self.split_type not in ['no_test']:
            test_perts = np.random.choice(combo_perts, int(len(combo_perts) * test_size))

        train_perts = [p for p in pert_list if p not in test_perts and p not in hold_out]
        return train_perts, list(test_perts)

    def _handle_combo_split(self, combo_perts, test_pert_genes, hold_outs):
        test_perts, hold_out = [], []
        if self.seen == 0:
            single = self.get_perts_from_genes(test_pert_genes, combo_perts, 'single')
            combo = self.get_perts_from_genes(test_pert_genes, combo_perts, 'combo')
            if hold_outs:
                hold_out = [c for c in combo if len([g for g in c.split('+') if g not in test_pert_genes]) > 0]
            test_perts = single + [c for c in combo if c not in hold_out]
        elif self.seen == 1:
            single = self.get_perts_from_genes(test_pert_genes, combo_perts, 'single')
            combo = self.get_perts_from_genes(test_pert_genes, combo_perts, 'combo')
            if hold_outs:
                hold_out = [c for c in combo if len([g for g in c.split('+') if g not in test_pert_genes]) > 1]
            test_perts = single + [c for c in combo if c not in hold_out]
        elif self.seen == 2:
            test_perts = np.random.choice(combo_perts, int(len(combo_perts) * 0.1))
        return test_perts

    def get_perts_from_genes(self, genes, pert_list, type_='both'):
        perts = []
        candidate_list = {
            'single': [p for p in pert_list if 'ctrl' in p and p != 'ctrl'],
            'combo': [p for p in pert_list if 'ctrl' not in p],
            'both': pert_list
        }[type_]

        for p in candidate_list:
            if any(g in parse_any_pert(p) for g in genes):
                perts.append(p)
        return perts

    def get_genes_from_perts(self, perts):
        if isinstance(perts, str):
            perts = [perts]
        genes = [g for p in np.unique(perts) for g in p.split('+') if g != 'ctrl']
        return np.unique(genes)
