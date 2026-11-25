import numpy as np, umap
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler
for name, loader in [("breast_cancer", load_breast_cancer), ("wine", load_wine)]:
    d = loader(); X = StandardScaler().fit_transform(d.data); y = d.target
    _, ro, re = umap.UMAP(densmap=True, output_dens=True, random_state=0).fit_transform(X)
    _, rou, reu = umap.UMAP(output_dens=True, random_state=0).fit_transform(X)
    print(name, X.shape, "rho densMAP=%.2f UMAP=%.2f" % (spearmanr(ro, re)[0], spearmanr(rou, reu)[0]))
    for c in np.unique(y):
        print("  ", d.target_names[c], "n=%d median r_orig=%.2f r_densmap=%.2f r_umap=%.2f" % ((y==c).sum(), np.median(ro[y==c]), np.median(re[y==c]), np.median(reu[y==c])))
