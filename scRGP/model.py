import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dims, norm=True, final_act=None):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 中间层
                if norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
        if final_act == "relu":
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PerturbationNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gene_num = args["num_genes"]
        self.pert_num = args["num_perts"]
        self.hidden = args["hidden_size"]
        self.device = args["device"]
        self.no_perturb = args["no_perturb"]

        # Embedding
        self.gene_lookup = nn.Embedding(self.gene_num, self.hidden, max_norm=True)
        self.pert_lookup = nn.Embedding(self.pert_num, self.hidden, max_norm=True)

        # Feature encoders
        self.gene_encoder = FeedForward([self.hidden, self.hidden, self.hidden], norm=True, final_act="relu")
        self.pert_encoder = FeedForward([self.hidden, self.hidden, self.hidden], norm=True, final_act="relu")
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden * 2, self.hidden),
            nn.ReLU()
        )

        # Decoders
        self.reconstruct = FeedForward([self.hidden, self.hidden * 2, self.hidden])
        self.cross_gene = FeedForward([self.gene_num, self.hidden, self.hidden])

        # Gene-specific params
        self.gene_weight1 = nn.Parameter(torch.empty(self.gene_num, self.hidden, 1))
        self.gene_bias1 = nn.Parameter(torch.empty(self.gene_num, 1))
        self.gene_weight2 = nn.Parameter(torch.empty(1, self.gene_num, self.hidden + 1))
        self.gene_bias2 = nn.Parameter(torch.empty(1, self.gene_num))
        nn.init.xavier_normal_(self.gene_weight1)
        nn.init.xavier_normal_(self.gene_bias1)
        nn.init.xavier_normal_(self.gene_weight2)
        nn.init.xavier_normal_(self.gene_bias2)

        # Normalizations
        self.bn_gene = nn.BatchNorm1d(self.hidden)
        self.bn_base = nn.BatchNorm1d(self.hidden)

    def forward(self, batch):
        x, pert_idx = batch.x, batch.pert_idx

        # Undisturbed mode: Return the input directly
        if self.no_perturb:
            return torch.stack(torch.split(x.view(-1, 1).flatten(), self.gene_num))

        n_graphs = batch.batch.max().item() + 1
        gene_ids = torch.arange(self.gene_num, device=self.device).repeat(n_graphs)

        g_emb = self.gene_lookup(gene_ids)
        g_emb = self.bn_gene(g_emb)
        g_emb = self.gene_encoder(g_emb).view(n_graphs, self.gene_num, -1)

        valid_mask = pert_idx != -1
        if valid_mask.any():
            pert_flat = pert_idx[valid_mask]
            p_emb = self.pert_lookup(pert_flat)
            p_emb = self.pert_encoder(p_emb)

            scatter_index = valid_mask.nonzero(as_tuple=False)[:, 0]
            fused = torch.zeros((n_graphs, self.hidden), device=self.device)
            fused.index_add_(0, scatter_index, p_emb)

            fusion_expanded = fused.unsqueeze(1).expand(-1, self.gene_num, -1)
            g_emb = self.fusion_layer(torch.cat([g_emb, fusion_expanded], dim=-1))

        g_flat = self.bn_base(g_emb.reshape(-1, self.hidden))
        decoded = self.reconstruct(F.relu(g_flat)).view(n_graphs, self.gene_num, -1)
        out = (decoded.unsqueeze(-1) * self.gene_weight1).sum(2) + self.gene_bias1
        out = out + x.view(out.shape)

        cg_feat = self.cross_gene(out.squeeze(2))
        cg_feat = cg_feat.repeat(1, self.gene_num).view(n_graphs, self.gene_num, -1)
        out = torch.cat([out, cg_feat], dim=2)
        out = (out * self.gene_weight2).sum(2) + self.gene_bias2

        out = out.view(n_graphs * self.gene_num, -1) + x.view(-1, 1)
        out = torch.stack(torch.split(out.flatten(), self.gene_num))
        return out
