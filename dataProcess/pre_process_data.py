import sys
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse, csr_matrix

warnings.filterwarnings("ignore")


def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    n_genes=200,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False
):
    """
    Calculate the differentially expressed genes under different
    covariate conditions and save them to adata.uns
    """
    gene_dict = {}
    for cov_cat in adata.obs[covariate].unique():
        ctrl_group_cov = f"{cov_cat}_{control_group}"
        adata_cov = adata[adata.obs[covariate] == cov_cat]

        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=ctrl_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            use_raw=False,
        )

        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict
    return gene_dict if return_dict else None


def get_DE_genes(adata, skip_calc_de=False):
    adata.obs = adata.obs.astype("category")
    if not skip_calc_de:
        rank_genes_groups_by_cov(
            adata,
            groupby="condition_name",
            covariate="cell_type",
            control_group="ctrl_1",
            n_genes=len(adata.var),
            key_added="rank_genes_groups_cov_all",
        )
    return adata


def get_dropout_non_zero_genes(adata):
    """
    Based on the average expression of ctrl and conditional samples,
    non-zero genes and non-dropout genes are extracted
    """
    condition2mean = {
        cond: np.mean(adata.X[adata.obs.condition == cond], axis=0)
        for cond in adata.obs.condition.unique()
    }

    pert_list = np.array(list(condition2mean.keys()))
    mean_expr = np.array(list(condition2mean.values())).reshape(len(pert_list), adata.X.shape[1])
    ctrl = mean_expr[np.where(pert_list == "ctrl")[0]]

    pert2pert_full_id = dict(adata.obs[["condition", "condition_name"]].values)
    pert_full_id2pert = dict(adata.obs[["condition_name", "condition"]].values)

    gene_id2idx = {g: i for i, g in enumerate(adata.var.index)}
    gene_idx2id = {i: g for i, g in enumerate(adata.var.index)}

    non_zeros_gene_idx = {}
    non_dropout_gene_idx = {}
    top_non_dropout_de_20 = {}
    top_non_zero_de_20 = {}

    for pert in adata.uns["rank_genes_groups_cov_all"].keys():
        cond = pert_full_id2pert[pert]
        X = np.mean(adata[adata.obs.condition == cond].X, axis=0)

        non_zero = np.where(np.array(X)[0] != 0)[0]
        zero = np.where(np.array(X)[0] == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        top_genes = adata.uns["rank_genes_groups_cov_all"][pert]
        top_gene_idx = [gene_id2idx[g] for g in top_genes]

        non_dropout_20 = [i for i in top_gene_idx if i in non_dropouts][:20]
        non_zero_20 = [i for i in top_gene_idx if i in non_zero][:20]

        non_zeros_gene_idx[pert] = np.sort(non_zero)
        non_dropout_gene_idx[pert] = np.sort(non_dropouts)
        top_non_dropout_de_20[pert] = np.array([gene_idx2id[i] for i in non_dropout_20])
        top_non_zero_de_20[pert] = np.array([gene_idx2id[i] for i in non_zero_20])

    adata.uns.update({
        "top_non_dropout_de_20": top_non_dropout_de_20,
        "non_dropout_gene_idx": non_dropout_gene_idx,
        "non_zeros_gene_idx": non_zeros_gene_idx,
        "top_non_zero_de_20": top_non_zero_de_20,
    })
    return adata


def extre_hvg(adata):
    """
    Extract HVG and forcibly retain perturbated genes on the basis of HVG
    """
    perturbed_genes = adata.obs["condition"].unique()
    perturbed_genes_unique = get_genes_from_perts(perturbed_genes)

    missing_genes = [g for g in perturbed_genes_unique if g not in adata.var["gene_name"].tolist()]
    print(f"perturbations are not in all genes：{missing_genes}")

    removed_pert = missing_list(missing_genes, perturbed_genes)
    adata = adata[~adata.obs["condition"].isin(removed_pert)]
    print("removed_pert:", removed_pert)

    sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="seurat_v3")
    hvg_genes = adata.var.loc[adata.var["highly_variable"], "gene_name"].tolist()

    missing_pert_genes = [g for g in perturbed_genes_unique if g not in hvg_genes]
    print(f"perturbations were not selected as HVG：{missing_pert_genes}")

    adata.var["highly_variable"] |= adata.var["gene_name"].isin(missing_pert_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()

    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    return adata


def data_norm(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def get_genes_from_perts(perts):
    if isinstance(perts, str):
        perts = [perts]
    genes = [g for p in np.unique(perts) for g in p.split("+") if g != "ctrl"]
    return list(np.unique(genes))


def missing_list(miss_list, perturbed_genes):
    removed = []
    for item in perturbed_genes:
        if item in miss_list:
            removed.append(item)
        elif "+" in item and any(p in miss_list for p in item.split("+")):
            removed.append(item)
    return removed


def filter_cell(adata, min_cells=30):
    """
    Remove perturbations with a single perturbation number less than min_cells
    """
    valid_conditions = adata.obs["condition"].value_counts()[lambda x: x >= min_cells].index
    adata = adata[adata.obs["condition"].isin(valid_conditions)].copy()
    adata.X = csr_matrix(adata.X)
    return adata


def format_norm(adata):
    """
    Format adata.obs and standardize the condition
    """
    def modify_condition(cond):
        parts = cond.split("+")
        if len(parts) != 2 and "ctrl" not in parts:
            if len(parts) == 1:
                parts.append("ctrl")
        return "+".join(parts)

    adata.obs["condition"] = adata.obs["condition"].apply(modify_condition)

    adata.obs = pd.DataFrame(adata.obs.values, index=adata.obs.index, columns=adata.obs.columns)
    adata.obs[["dose_val", "control"]] = adata.obs["condition"].apply(
        lambda x: ("1", "1") if x == "ctrl" else ("1+1+1", "0")
    ).apply(pd.Series)

    adata.obs["condition_name"] = (
        adata.obs["cell_type"].astype(str) + "_" +
        adata.obs["condition"].astype(str) + "_" +
        adata.obs["dose_val"].astype(str)
    )

    if "batch" in adata.obs:
        del adata.obs["batch"]
    return adata


if __name__ == "__main__":
    read_file, write_file = sys.argv[1], sys.argv[2]
    adata = sc.read_h5ad(read_file)
    # Extract highly variable genes
    adata = extre_hvg(adata)
    # Data standardization
    adata = data_norm(adata)
    # Format standardization
    adata = format_norm(adata)
    # Calculate the differential expression set
    adata = get_DE_genes(adata, skip_calc_de=False)
    adata = get_dropout_non_zero_genes(adata)

    adata.X = csr_matrix(adata.X)
    adata.write_h5ad(write_file)
