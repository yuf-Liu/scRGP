import os
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from model import PerturbationNet
from utils import loss_fct, create_cell_graph_dataset_for_prediction
from inference import evaluate, compute_acc, compute_metrics, print_result


class scRGP:
    def __init__(self, pert_data, device='cuda'):
        self.device = device
        self.config = None
        self.dataloader = pert_data.dataloader
        self.gene_list = pert_data.gene_names
        self.pert_list = pert_data.pert_names
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.adata = pert_data.adata

        self.ctrl_expression = torch.tensor(
            np.mean(self.adata.X[self.adata.obs.condition == 'ctrl'], axis=0)
        ).to(self.device)

        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {
            pert_full_id2pert[i]: j
            for i, j in self.adata.uns['non_zeros_gene_idx'].items()
            if i in pert_full_id2pert
        }

    # ---------------- Model ---------------- #
    def model_initialize(self, hidden_size=64, decoder_hidden_size=16,
                         direction_lambda=1e-1, no_perturb=False, **kwargs):
        self.config = {
            'hidden_size': hidden_size,
            'decoder_hidden_size': decoder_hidden_size,
            'direction_lambda': direction_lambda,
            'device': self.device,
            'num_genes': self.num_genes,
            'num_perts': self.num_perts,
            'no_perturb': no_perturb
        }
        self.model = PerturbationNet(self.config).to(self.device)
        self.best_model = deepcopy(self.model)

    # ---------------- Training ---------------- #
    def train(self, epochs=20, lr=1e-3, weight_decay=5e-4):
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader.get('val_loader')

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = float('inf')

        print('Start Training...')
        for epoch in range(epochs):
            self.model.train()
            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch)
                loss = loss_fct(pred, batch.y, batch.pert,
                                ctrl=self.ctrl_expression,
                                dict_filter=self.dict_filter,
                                direction_lambda=self.config['direction_lambda'])
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
                optimizer.step()
                if step % 50 == 0:
                    print(f"Epoch {epoch+1} Step {step+1} Loss: {loss.item():.4f}")

            scheduler.step()
            self._evaluate_epoch(epoch, train_loader, val_loader, min_val)

        print("Training Finished")
        self._test()

    def _evaluate_epoch(self, epoch, train_loader, val_loader, min_val):
        train_res = evaluate(train_loader, self.model, self.device, self.config['no_perturb'])
        train_metrics, _ = compute_metrics(train_res)
        print(f"Epoch {epoch+1} Train Acc (Top20-200): {compute_acc(train_res)}")
        print_result(train_metrics)

        if val_loader:
            val_res = evaluate(val_loader, self.model, self.device, self.config['no_perturb'])
            val_metrics, _ = compute_metrics(val_res)
            print_result(val_metrics)
            if val_metrics['mse_de_20'] < min_val:
                min_val = val_metrics['mse_de_20']
                self.best_model = deepcopy(self.model)

    def _test(self):
        if 'test_loader' not in self.dataloader:
            print("No test data available")
            return
        test_loader = self.dataloader['test_loader']
        test_res = evaluate(test_loader, self.best_model, self.device, self.config['no_perturb'])
        test_metrics, _ = compute_metrics(test_res)
        print("Test Performance:")
        print(f"Acc (Top20-200): {compute_acc(test_res)}")
        print_result(test_metrics)

    # ---------------- Model I/O ---------------- #
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))

    def load_pretrained(self, path):
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        del config['device'], config['num_genes'], config['num_perts']
        self.model_initialize(**config)
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.best_model = self.model.to(self.device)

    # ---------------- Prediction ---------------- #
    def predict(self, adata_file, pert_list=None, save_file=None):
        _adata = sc.read_h5ad(adata_file)
        ctrl_adata = _adata[_adata.obs['condition'] == 'ctrl']

        pert_list = pert_list or _adata.var['gene_name']

        results_pred = {}
        self.best_model.eval()

        from torch_geometric.data import DataLoader
        for pert in pert_list:
            cg = create_cell_graph_dataset_for_prediction(pert, ctrl_adata,
                                                          self.gene_list, self.device, num_samples=300)
            loader = DataLoader(cg, ctrl_adata.shape[0], shuffle=False)
            batch = next(iter(loader)).to(self.device)
            with torch.no_grad():
                p = self.best_model(batch).cpu().numpy()
            results_pred[str(pert)] = np.mean(p, axis=0)

        pred_matrix = np.vstack(list(results_pred.values()))
        pred_obs = pd.DataFrame({'condition': list(results_pred.keys())})
        pred_adata = sc.AnnData(X=pred_matrix, obs=pred_obs, var=ctrl_adata.var.copy())

        if save_file:
            pred_adata.write(save_file)
        return pred_adata
