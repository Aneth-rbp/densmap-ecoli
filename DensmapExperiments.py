"""densMAP vs UMAP vs PCA with the methodology of Narayan, Berger & Cho (2021) (Proyecto U2).

Parameters and figures follow densvis/densmap/trial_densmap.py: n_neighbors=30, n_epochs=750,
dens_frac=0.3, dens_lambda=2, var_shift=0.1; UMAP is the same code with dens_lambda=0 and
dens_frac=0; density preservation is the correlation between the log local radii ro and re
returned by the algorithm. Runs on umap-learn, where the authors' densMAP code was merged.

Usage: python DensmapExperiments.py OUT_DIR TRIAL_DATA [--seeds 5] [--backend umap-learn|authors]
                                    [--steps ... real --real-data sae-ecoli-dna/data]
--backend authors runs the authors' own densmap-learn package (densvis repo) instead.
TRIAL_DATA is densvis/densmap/densmap/trial_data.txt (six Gaussians of increasing variance).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

PAPER = dict(n_neighbors=30, n_epochs=750, dens_frac=0.3, dens_lambda=2.0, dens_var_shift=0.1)


def MakeGaussianClusters(sigmas, n_per_cluster=500, dim=20, separation=30.0, seed=0):
    rng = np.random.default_rng(seed)
    points, labels = [], []
    for i, sigma in enumerate(sigmas):
        center = np.zeros(dim)
        center[i % dim] = separation * (1 + i // dim)
        points.append(center + sigma * rng.standard_normal((n_per_cluster, dim)))
        labels.append(np.full(n_per_cluster, i))
    return np.vstack(points), np.concatenate(labels)


def LoadTrial(path):
    X = np.loadtxt(path)
    return X, np.arange(len(X)) // (len(X) // 6)


def LogRadiusKnn(X, k):
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return np.log(distances[:, 1:].mean(axis=1))


AUTHORS_DENSMAP = None


def LoadAuthorsDensmap():
    """The authors' densmap-learn, patched only to run on numba 0.50 (aarch64 has no numba 0.48)."""
    import sys
    import numba
    import numba.core.errors
    sys.modules["numba.errors"] = numba.core.errors
    import densmap
    import densmap.densmap_ as module
    # parallel=True miscompiles on numba 0.50 aarch64 (ZeroDivisionError, NaN); the function has no prange
    module.optimize_layout = numba.njit(fastmath=True)(module.optimize_layout.py_func)
    return densmap.densMAP


def Embed(method, X, seed, **overrides):
    """Returns (embedding, ro, re). PCA has no ro/re of its own, so it gets a kNN radius."""
    params = {**PAPER, **overrides}
    if method == "PCA":
        Y = PCA(2, random_state=seed).fit_transform(X)
        return Y, LogRadiusKnn(X, params["n_neighbors"]), LogRadiusKnn(Y, params["n_neighbors"])
    if method == "UMAP":
        params.update(dens_lambda=0.0, dens_frac=0.0)
    if AUTHORS_DENSMAP is not None:
        params["var_shift"] = params.pop("dens_var_shift")
        return AUTHORS_DENSMAP(n_components=2, logdist_shift=0, final_dens=True, random_state=seed,
                               **params).fit_transform(X)
    return umap.UMAP(densmap=True, output_dens=True, random_state=seed, compatibility_layout=True,
                     **params).fit_transform(X)


def ClusterSpread(Y, labels):
    return {c: np.median(np.linalg.norm(Y[labels == c] - Y[labels == c].mean(0), axis=1)) for c in np.unique(labels)}


def Evaluate(X, Y, ro, re, labels):
    return {
        "density_pearson": pearsonr(ro, re)[0],
        "density_spearman": spearmanr(ro, re)[0],
        "trustworthiness": trustworthiness(X, Y, n_neighbors=PAPER["n_neighbors"]),
        "knn_accuracy": cross_val_score(KNeighborsClassifier(15), Y, labels, cv=5).mean(),
    }


def Summarize(rows, keys):
    table = pd.DataFrame(rows)
    metrics = [c for c in table.columns if c not in keys + ["seed"]]
    return table.groupby(keys)[metrics].agg(["mean", "std"]).round(3)


def SpreadColumns(Y, X, labels):
    spread, original = ClusterSpread(Y, labels), ClusterSpread(X, labels)
    last = max(spread)
    return {"spread_ratio_max_to_min": spread[last] / spread[0],
            "original_ratio_max_to_min": original[last] / original[0]}


def RunClusters(name, X_for_seed, names, out_dir, seeds):
    rows, examples = [], {}
    for seed in seeds:
        X, labels = X_for_seed(seed)
        for method in ["PCA", "UMAP", "densMAP"]:
            Y, ro, re = Embed(method, X, seed)
            rows.append({"method": method, "seed": seed, **Evaluate(X, Y, ro, re, labels),
                         **SpreadColumns(Y, X, labels)})
            if seed == seeds[0]:
                examples[method] = (Y, ro, re)
    Save(rows, ["method"], out_dir, name)
    PlotPaperFigure(examples, labels, names, os.path.join(out_dir, f"{name}_figure.png"))


def RunSyntheticSweeps(out_dir, seeds):
    """Where densMAP stops working: hyperparameters and harder data."""
    scenarios = {
        "base 1:2:4": dict(sigmas=[1, 2, 4]),
        "extreme 1:4:16": dict(sigmas=[1, 4, 16]),
        "overlap sep=8": dict(sigmas=[1, 2, 4], separation=8.0),
        "dim=200": dict(sigmas=[1, 2, 4], dim=200, separation=60.0),
        "unbalanced n": dict(sigmas=[1, 2, 4], n_per_cluster=100),
    }
    rows = []
    for seed in seeds:
        for scenario, params in scenarios.items():
            X, labels = MakeGaussianClusters(seed=seed, **params)
            for method in ["UMAP", "densMAP"]:
                Y, ro, re = Embed(method, X, seed)
                rows.append({"sweep": "scenario", "setting": scenario, "method": method, "seed": seed,
                             **Evaluate(X, Y, ro, re, labels), **SpreadColumns(Y, X, labels)})
        X, labels = MakeGaussianClusters([1, 2, 4], seed=seed)
        for sweep, values in [("dens_lambda", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]), ("n_neighbors", [5, 15, 30, 50, 150])]:
            for value in values:
                Y, ro, re = Embed("densMAP", X, seed, **{sweep: value})
                rows.append({"sweep": sweep, "setting": value, "method": "densMAP", "seed": seed,
                             **Evaluate(X, Y, ro, re, labels), **SpreadColumns(Y, X, labels)})
    Save(rows, ["sweep", "setting", "method"], out_dir, "synthetic_sweeps")


def RunBenchmark(out_dir, seeds):
    data = load_breast_cancer()
    X, labels = StandardScaler().fit_transform(data.data), data.target
    rows, examples = [], {}
    for seed in seeds:
        for method in ["PCA", "UMAP", "densMAP"]:
            Y, ro, re = Embed(method, X, seed)
            row = {"method": method, "seed": seed, **Evaluate(X, Y, ro, re, labels)}
            for c, name in enumerate(data.target_names):
                row[f"median_ro_{name}"] = np.median(ro[labels == c])
                row[f"median_re_{name}"] = np.median(re[labels == c])
            rows.append(row)
            if seed == seeds[0]:
                examples[method] = (Y, ro, re)
    Save(rows, ["method"], out_dir, "benchmark")
    PlotPaperFigure(examples, labels, list(data.target_names), os.path.join(out_dir, "benchmark_figure.png"))


FUNCTIONAL_GROUPS = [
    ("ribosome", {"GO:0005840"}),
    ("flagellar motility", {"GO:0071973"}),
    ("amino acid biosynthesis", {"GO:0008652"}),
    ("DNA replication/repair", {"GO:0006260", "GO:0006281"}),
    ("transcription regulation", {"GO:0006355"}),
    ("carbohydrate metabolism", {"GO:0005975"}),
    ("transmembrane transport", {"GO:0055085"}),
]


def LoadEcoli(data_dir):
    """Evo 2 gene embeddings (sae-ecoli-dna), z-scored per dimension as the SAE repo does."""
    X = np.load(os.path.join(data_dir, "embeddings", "GeneEmbeddings.npy")).astype(np.float64)
    ids = open(os.path.join(data_dir, "embeddings", "GeneIds.txt")).read().split()
    genes = pd.read_csv(os.path.join(data_dir, "processed", "Genes.tsv"), sep="\t",
                        usecols=["locusTag", "geneName", "cdsLength", "goTerms"]).set_index("locusTag").loc[ids]
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - X.mean(axis=0)) / std
    names = ["other"] + [name for name, _ in FUNCTIONAL_GROUPS]
    codes = []
    for terms in genes["goTerms"].fillna("").str.split(";"):
        match = [i + 1 for i, (_, group) in enumerate(FUNCTIONAL_GROUPS) if group & set(terms)]
        codes.append(match[0] if match else 0)
    genes["group"] = [names[c] for c in codes]
    genes["goCount"] = genes["goTerms"].fillna("").str.count("GO:")
    return X, np.array(codes), names, genes.drop(columns="goTerms")


def RunReal(data_dir, out_dir, seeds):
    X, labels, names, genes = LoadEcoli(data_dir)
    annotated = labels > 0
    rows, examples = [], {}
    for seed in seeds:
        for method in ["PCA", "UMAP", "densMAP"]:
            Y, ro, re = Embed(method, X, seed)
            row = {"method": method, "seed": seed, **Evaluate(X, Y, ro, re, labels)}
            row["knn_accuracy"] = cross_val_score(KNeighborsClassifier(15), Y[annotated], labels[annotated], cv=5).mean()
            for column in ["cdsLength", "goCount"]:
                row[f"spearman_ro_{column}"] = spearmanr(ro, genes[column])[0]
                row[f"spearman_re_{column}"] = spearmanr(re, genes[column])[0]
            for c, name in enumerate(names):
                row[f"median_ro_{name}"] = np.median(ro[labels == c])
                row[f"median_re_{name}"] = np.median(re[labels == c])
            rows.append(row)
            if seed == seeds[0]:
                examples[method] = (Y, ro, re)
                genes[f"{method}_x"], genes[f"{method}_y"] = Y[:, 0], Y[:, 1]
                genes[f"{method}_ro"], genes[f"{method}_re"] = ro, re
    Save(rows, ["method"], out_dir, "real")
    genes.to_csv(os.path.join(out_dir, "real_genes_seed0.csv"))
    PlotPaperFigure(examples, labels, names, os.path.join(out_dir, "real_figure.png"), other_gray=True)
    PlotByLength(examples, genes["cdsLength"].to_numpy(), os.path.join(out_dir, "real_by_length.png"))


def PlotByLength(examples, lengths, path):
    fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 4.5))
    for ax, (method, (Y, _, _)) in zip(axes, examples.items()):
        points = ax.scatter(*Y.T, s=2, c=np.log10(lengths), cmap="viridis")
        ax.set_title(f"Embedding ({method})")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(points, ax=axes, label="log10 gene length (bp)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def Save(rows, keys, out_dir, name):
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{name}_raw.csv"), index=False)
    Summarize(rows, keys).to_csv(os.path.join(out_dir, f"{name}.csv"))


def PlotPaperFigure(examples, labels, names, path, other_gray=False):
    """One row per method: embedding and density preservation (ro vs re), as in trial_densmap.py."""
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    if other_gray:
        colors = [(0.8, 0.8, 0.8, 1.0)] + list(plt.cm.tab10(np.arange(len(names) - 1)))
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 5 * len(examples)))
    for row, (method, (Y, ro, re)) in zip(axes, examples.items()):
        for c, name in enumerate(names):
            mask = labels == c
            row[0].scatter(*Y[mask].T, s=2, alpha=0.4, color=colors[c], label=name)
            row[1].scatter(ro[mask], re[mask], s=2, alpha=0.4, color=colors[c])
        row[0].set_title(f"Embedding ({method})")
        row[0].set_xticks([])
        row[0].set_yticks([])
        row[1].set_title(f"Density Preservation ({method}), r = {pearsonr(ro, re)[0]:.2f}")
        row[1].set_xlabel("Original Local Radius (log)")
        row[1].set_ylabel("Embedding Local Radius (log)")
    axes[0][0].legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("trial_data")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--backend", choices=["umap-learn", "authors"], default="umap-learn")
    parser.add_argument("--steps", nargs="+", default=["trial", "synthetic", "sweeps", "benchmark"],
                        choices=["trial", "synthetic", "sweeps", "benchmark", "real"])
    parser.add_argument("--real-data", help="sae-ecoli-dna/data directory, needed by the real step")
    args = parser.parse_args()
    global AUTHORS_DENSMAP, umap
    if args.backend == "authors":
        AUTHORS_DENSMAP = LoadAuthorsDensmap()
    else:
        import umap
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = list(range(args.seeds))
    sigmas = [1, 2, 4]
    steps = [
        ("trial", lambda: RunClusters("trial", lambda seed: LoadTrial(args.trial_data),
                                      [f"cluster {i + 1}" for i in range(6)], args.out_dir, seeds)),
        ("synthetic", lambda: RunClusters("synthetic", lambda seed: MakeGaussianClusters(sigmas, seed=seed),
                                          [f"σ={s}" for s in sigmas], args.out_dir, seeds)),
        ("sweeps", lambda: RunSyntheticSweeps(args.out_dir, seeds)),
        ("benchmark", lambda: RunBenchmark(args.out_dir, seeds)),
        ("real", lambda: RunReal(args.real_data, args.out_dir, seeds)),
    ]
    for name, step in [(name, step) for name, step in steps if name in args.steps]:
        print("running", name, flush=True)
        step()
    print("done", flush=True)


if __name__ == "__main__":
    Main()
