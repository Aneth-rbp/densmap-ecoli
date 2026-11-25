"""UMAP vs densMAP side by side on the same axes scale, with each group's size relative to the first group.

Usage: python PlotSideBySide.py DATASET OUT_PNG [--seed 0] [--trial-data PATH] [--real-data DIR]
DATASET is one of synthetic, trial, benchmark, real.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

import DensmapExperiments as experiments

SIGMAS = [1, 2, 4]


def LoadDataset(dataset, seed, trial_data, real_data):
    if dataset == "synthetic":
        X, labels = experiments.MakeGaussianClusters(SIGMAS, seed=seed)
        return X, labels, [f"σ={s}" for s in SIGMAS]
    if dataset == "trial":
        X, labels = experiments.LoadTrial(trial_data)
        return X, labels, [f"cluster {i + 1}" for i in range(6)]
    if dataset == "benchmark":
        data = load_breast_cancer()
        return StandardScaler().fit_transform(data.data), data.target, list(data.target_names)
    X, labels, names, _ = experiments.LoadEcoli(real_data)
    return X, labels, names


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["synthetic", "trial", "benchmark", "real"])
    parser.add_argument("out_png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trial-data")
    parser.add_argument("--real-data")
    args = parser.parse_args()
    experiments.AUTHORS_DENSMAP = experiments.LoadAuthorsDensmap()
    X, labels, names = LoadDataset(args.dataset, args.seed, args.trial_data, args.real_data)
    original = experiments.ClusterSpread(X, labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    if args.dataset == "real":
        colors = [(0.8, 0.8, 0.8, 1.0)] + list(plt.cm.tab10(np.arange(len(names) - 1)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    for ax, method in zip(axes, ["UMAP", "densMAP"]):
        Y, _, _ = experiments.Embed(method, X, args.seed)
        Y = Y - Y.mean(axis=0)
        spread = experiments.ClusterSpread(Y, labels)
        for c, name in enumerate(names):
            ax.scatter(*Y[labels == c].T, s=3, alpha=0.5, color=colors[c],
                       label=f"{name}: {spread[c] / spread[0]:.1f}× (original {original[c] / original[0]:.1f}×)")
        ax.set_title(method)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=8, markerscale=4)
    fig.suptitle(f"{args.dataset}: same axes scale; group size relative to {names[0]}")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    Main()
