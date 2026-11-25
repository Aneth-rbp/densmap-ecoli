import sys, numpy as np, pandas as pd, umap
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
root = sys.argv[1]
X = np.load(root + "/data/embeddings/GeneEmbeddings.npy")
ids = open(root + "/data/embeddings/GeneIds.txt").read().split()
g = pd.read_csv(root + "/data/processed/Genes.tsv", sep="\t").set_index("locusTag").loc[ids]
Xp = PCA(50, random_state=0).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-8))
emb, ro, re = umap.UMAP(densmap=True, output_dens=True, random_state=0).fit_transform(Xp)
print("rho(log r_orig, log r_emb) =", spearmanr(ro, re)[0])
_, ro_u, re_u = umap.UMAP(output_dens=True, random_state=0).fit_transform(Xp)
print("UMAP rho =", spearmanr(ro_u, re_u)[0])
seq = g["sequence"].str.upper()
gc = seq.apply(lambda s: (s.count("G") + s.count("C")) / len(s))
print("rho(r_orig, GC) =", spearmanr(ro, gc)[0], " rho(r_orig, cdsLength) =", spearmanr(ro, g["cdsLength"])[0])
go = g["goTerms"].fillna("")
for name, term in [("ribosome struct", "GO:0003735"), ("transposase", "GO:0004803"), ("transport", "GO:0006810"), ("DNA-binding TF", "GO:0003700")]:
    m = go.str.contains(term).values
    p = mannwhitneyu(ro[m], ro[~m])[1]
    print(f"{name:16s} n={m.sum():4d} median r_orig in={np.median(ro[m]):.3f} out={np.median(ro[~m]):.3f} p={p:.2e}")
yg = g["geneName"].str.match(r"^y[a-z]{3}").values
print(f"y-genes n={yg.sum()} median in={np.median(ro[yg]):.3f} out={np.median(ro[~yg]):.3f} p={mannwhitneyu(ro[yg], ro[~yg])[1]:.2e}")
