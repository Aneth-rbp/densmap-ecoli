"""Figuras del artículo, leídas solo de resultados_autores/ (no recalcula embeddings).

Uso: python figuras/HacerFiguras.py   (escribe en figuras/)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
RES = os.path.normpath(os.path.join(HERE, "..", "resultados_autores"))

COLUMN = 3.5
DOUBLE = 7.16
COLOR = {"UMAP": "#404040", "densMAP": "#1f5fa8", "PCA": "#e08214", "original": "black"}
GROUPS = [  # mismo orden que FUNCTIONAL_GROUPS en DensmapExperiments.py
    ("ribosome", "ribosoma"),
    ("flagellar motility", "motilidad flagelar"),
    ("amino acid biosynthesis", "biosíntesis de aminoácidos"),
    ("DNA replication/repair", "replicación/reparación de ADN"),
    ("transcription regulation", "regulación de la transcripción"),
    ("carbohydrate metabolism", "metabolismo de carbohidratos"),
    ("transmembrane transport", "transporte transmembrana"),
]
GROUP_NAME = dict(GROUPS) | {"other": "otros"}
GROUP_COLOR = {g: plt.cm.tab10(i) for i, (g, _) in enumerate(GROUPS)} | {"other": (0.8, 0.8, 0.8, 1.0)}
COPIES = "#7b2d8e"  # copias idénticas: color fuera de tab10 y de los métodos

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.titlesize": 8, "axes.titlelocation": "left",
})
PREVIEW = os.environ.get("PREVIEW")  # carpeta opcional para copias PNG de revisión
NOTE = dict(fontsize=6.5, color="0.35", style="italic")


def ReadSummary(name, index):
    return pd.read_csv(os.path.join(RES, name), header=[0, 1], index_col=list(range(index)))


def Title(ax, letter, text, pad=4, x=0.0):
    ax.set_title(f"({letter}) {text}", pad=pad, x=x)


def Save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), **kw)
    if PREVIEW:
        fig.savefig(os.path.join(PREVIEW, os.path.splitext(name)[0] + ".png"), dpi=200,
                    **{k: v for k, v in kw.items() if k != "dpi"})


def Sintetico():
    """Recompone synthetic_side_by_side.png (semilla 0): recorta el área de cada eje, borra la leyenda en
    inglés y redibuja ejes y leyenda en español. Píxeles del marco y marcas medidos sobre el PNG original."""
    img = np.array(Image.open(os.path.join(RES, "synthetic_side_by_side.png")).convert("RGB"))
    top, bottom = 210, 1042
    px_per_unit = 174.5 / 5
    # tamaños relativos a σ=1 tal como los imprime la leyenda del PNG (semilla 0)
    sizes = {"UMAP": ["1.0", "1.0", "1.0"], "densMAP": ["1.0", "1.9", "3.1"]}
    original = ["1.0", "2.0", "4.0"]
    viridis = plt.cm.viridis(np.linspace(0, 1, 3))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.75), sharey=True)
    for ax, (method, x0, letter) in zip(axes, [("UMAP", 99, "a"), ("densMAP", 1250, "b")]):
        x1 = x0 + 1109
        tick15 = 187.5 + (x0 - 99)
        crop = img[top:bottom, x0:x1].copy()
        crop[219 - top:344 - top, 107 - 99:582 - 99] = 255
        extent = [-15 + (x0 - tick15) / px_per_unit, -15 + (x1 - tick15) / px_per_unit,
                  10 - (bottom - 342.5) / px_per_unit, 10 + (342.5 - top) / px_per_unit]
        ax.imshow(crop, extent=extent, interpolation="antialiased")
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(True)
        handles = [Line2D([], [], ls="", marker="o", ms=4, color=viridis[i],
                          label=f"σ = {s}: {sizes[method][i]}×  ·  original {original[i]}×")
                   for i, s in enumerate([1, 2, 4])]
        ax.legend(handles=handles, loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.3,
                  title="Tamaño relativo a σ = 1", title_fontproperties=dict(weight="bold", size=7),
                  alignment="left")
        Title(ax, letter, f"Embedding {method} del sintético")
        ax.set_xlabel("Dimensión 1 del embedding")
    axes[0].set_ylabel("Dimensión 2 del embedding")
    fig.tight_layout(w_pad=1.0)
    Save(fig, "sintetico_umap_densmap.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def Sensibilidad():
    s = ReadSummary("synthetic_sweeps.csv", 3)
    lam = s.loc["dens_lambda"].xs("densMAP", level="method")
    x = lam.index.astype(float).to_numpy()
    umap = s.loc[("scenario", "base 1:2:4", "UMAP")]
    orig = s.loc[("scenario", "base 1:2:4", "densMAP")]
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 3.7), sharex=True)
    ax = axes[0]
    ax.axhline(orig[("original_ratio_max_to_min", "mean")], color=COLOR["original"], ls=":", lw=1,
               label=f"original = {orig[('original_ratio_max_to_min', 'mean')]:.2f}")
    ax.axhline(umap[("spread_ratio_max_to_min", "mean")], color=COLOR["UMAP"], lw=1,
               label=f"UMAP = {umap[('spread_ratio_max_to_min', 'mean')]:.2f}")
    ax.errorbar(x, lam[("spread_ratio_max_to_min", "mean")], yerr=lam[("spread_ratio_max_to_min", "std")],
                color=COLOR["densMAP"], marker="o", ms=3.5, lw=1.2, capsize=2, label="densMAP")
    ax.set_ylabel("Razón de dispersión")
    ax.text(0.98, 0.04, "σ = 4 / σ = 1", transform=ax.transAxes, ha="right", va="bottom", **NOTE)
    ax.legend(frameon=False, loc="upper left")
    Title(ax, "a", "Separación de dispersiones según λ")
    ax = axes[1]
    ax.axhline(umap[("trustworthiness", "mean")], color=COLOR["UMAP"], lw=1,
               label=f"UMAP = {umap[('trustworthiness', 'mean')]:.3f}")
    ax.errorbar(x, lam[("trustworthiness", "mean")], yerr=lam[("trustworthiness", "std")],
                color=COLOR["densMAP"], marker="o", ms=3.5, lw=1.2, capsize=2, label="densMAP")
    ax.set_ylabel("Confiabilidad $\\mathbf{T}$")
    ax.set_ylim(0.868, 0.901)
    ax.set_xlabel("Peso de densidad λ")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.minorticks_off()
    ax.legend(frameon=False, loc="lower left")
    ax.text(0.62, 0.04, "eje x en escala log", transform=ax.transAxes, ha="center", va="bottom", **NOTE)
    for a in axes:
        a.axvline(2, color="0.85", lw=0.8, zorder=0)
        a.text(2.1, 0.97, "por defecto", transform=a.get_xaxis_transform(), ha="left", va="top", **NOTE)
    Title(ax, "b", "Confiabilidad del vecindario según λ")
    fig.tight_layout(h_pad=0.6)
    Save(fig, "sensibilidad_lambda.pdf")
    plt.close(fig)


def Reales():
    g = pd.read_csv(os.path.join(RES, "real_genes_seed0.csv"))
    keys = ["other"] + [k for k, _ in GROUPS]
    width, height = COLUMN, 6.75
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.3 / width, 4.1 / height, 3.15 / width, 2.35 / height])
    for name in keys:
        d = g[g.group == name]
        ax.scatter(d.densMAP_x, d.densMAP_y, s=1.5 if name == "other" else 2.5, lw=0,
                   color=GROUP_COLOR[name], alpha=0.6 if name == "other" else 0.85, rasterized=True)
    lo, hi = g[["densMAP_x", "densMAP_y"]].quantile(0.005), g[["densMAP_x", "densMAP_y"]].quantile(0.995)
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo.densMAP_x - pad.densMAP_x, hi.densMAP_x + pad.densMAP_x)
    ax.set_ylim(lo.densMAP_y - pad.densMAP_y, hi.densMAP_y + pad.densMAP_y)
    inside = g.densMAP_x.between(*ax.get_xlim()) & g.densMAP_y.between(*ax.get_ylim())
    print(f"Reales (a): {(~inside).sum()} genes fuera del recuadro (percentiles 0.5-99.5 + 5 %)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(True)
    ax.set_xlabel("Dimensión 1 de densMAP")
    ax.set_ylabel("Dimensión 2 de densMAP")
    Title(ax, "a", f"Embedding densMAP de {len(g)} genes de E. coli")
    handles = [Line2D([], [], ls="", marker="o", ms=4, color=GROUP_COLOR[k],
                      label=GROUP_NAME[k]) for k in keys]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 3.87 / height), frameon=False, ncol=2,
               handletextpad=0.05, borderaxespad=0, borderpad=0.1, labelspacing=0.3, columnspacing=0.5,
               title="Grupo funcional GO", title_fontproperties=dict(weight="bold", size=7), alignment="left")
    ax = fig.add_axes([1.25 / width, 0.45 / height, 2.2 / width, 2.2 / height])
    order = g.groupby("group").densMAP_ro.median().sort_values(ascending=False).index.tolist()
    pos = np.arange(len(order))
    box = dict(widths=0.34, vert=False, patch_artist=True, showfliers=False,
               medianprops=dict(color="black", lw=1), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    b1 = ax.boxplot([g.loc[g.group == k, "densMAP_re"] for k in order], positions=pos - 0.2, **box)
    b2 = ax.boxplot([g.loc[g.group == k, "UMAP_re"] for k in order], positions=pos + 0.2, **box)
    for patch, k in zip(b1["boxes"], order):
        patch.set(facecolor=GROUP_COLOR[k], edgecolor="black", lw=0.6)
    for patch in b2["boxes"]:
        patch.set(facecolor="white", edgecolor=COLOR["UMAP"], lw=0.6)
    ax.set_yticks(pos)
    wrap = {"replicación/reparación de ADN": "replicación/reparación\nde ADN",
            "regulación de la transcripción": "regulación de la\ntranscripción",
            "metabolismo de carbohidratos": "metabolismo de\ncarbohidratos",
            "biosíntesis de aminoácidos": "biosíntesis de\naminoácidos",
            "transporte transmembrana": "transporte\ntransmembrana"}
    ax.set_yticklabels([wrap.get(GROUP_NAME[k], GROUP_NAME[k]) for k in order], linespacing=0.95)
    ax.invert_yaxis()
    ax.set_xlabel("Log-radio local en el embedding, log $\\mathbf{r_e}$")
    for p, k in zip(pos, order):
        ax.text(0.99, p, f"n = {(g.group == k).sum()}", transform=ax.get_yaxis_transform(), ha="right",
                va="center", **NOTE)
    ax.legend(handles=[Patch(facecolor="0.6", edgecolor="black", lw=0.6, label="densMAP"),
                       Patch(facecolor="white", edgecolor=COLOR["UMAP"], lw=0.6, label="UMAP")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, borderaxespad=0.2,
              handlelength=1.2, columnspacing=0.8)
    ax.set_xlim(right=2.6)
    Title(ax, "b", "Log-radio local por grupo funcional", pad=16, x=-0.43)
    Save(fig, "reales_densmap.pdf", dpi=600)
    plt.close(fig)
    med = g.groupby("group")[["densMAP_ro", "densMAP_re", "UMAP_re", "cdsLength"]].median()
    print("Reales, semilla 0, medianas por grupo (orden de la figura):")
    print(med.loc[order].round(2).to_string())


def Falla():
    s = ReadSummary("synthetic_sweeps.csv", 3)
    t = ReadSummary("trial.csv", 1)
    names = [("base 1:2:4", "base\n1:2:4"), ("dim=200", "dim.\n200"), ("extreme 1:4:16", "extremo\n1:4:16"),
             ("overlap sep=8", "traslape\nsep. 8"), ("unbalanced n", "n redu-\ncido")]
    rows = [(lab, s.loc[("scenario", k, "UMAP")], s.loc[("scenario", k, "densMAP")]) for k, lab in names]
    rows.append(("6 gauss.\nde autores", t.loc["UMAP"], t.loc["densMAP"]))
    g = pd.read_csv(os.path.join(RES, "real_genes_seed0.csv"))
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 5.0))
    ax = axes[0]
    pos = np.arange(len(rows))
    mean, std = ("spread_ratio_max_to_min", "mean"), ("spread_ratio_max_to_min", "std")
    for off, i, m in [(-0.19, 1, "UMAP"), (0.19, 2, "densMAP")]:
        ax.bar(pos + off, [r[i][mean] for r in rows], yerr=[r[i][std] for r in rows], width=0.36,
               color=COLOR[m], label=m, error_kw=dict(lw=0.6, capsize=1.5))
    ax.scatter(pos, [r[2][("original_ratio_max_to_min", "mean")] for r in rows], marker="_", s=120,
               color=COLOR["original"], lw=1.6, zorder=3, label="original")
    ax.axvline(len(rows) - 1.5, color="0.7", lw=0.6, ls="--")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.minorticks_off()
    ax.set_ylabel("Razón de dispersión")
    ax.text(0.02, 0.82, "eje y en escala log", transform=ax.transAxes, ha="left", va="top", **NOTE)
    ax.set_xticks(pos)
    ax.set_xticklabels([r[0] for r in rows])
    ax.legend(frameon=False, loc="upper left", ncol=3)
    ax.set_ylim(0.7, 110)
    Title(ax, "a", "Razón de dispersión por escenario de falla")
    ax = axes[1]
    dup = g.duplicated(["PCA_x", "PCA_y"], keep=False)
    ax.scatter(g.cdsLength[~dup], g.densMAP_ro[~dup], s=2, lw=0, color=COLOR["original"], alpha=0.3,
               rasterized=True, label="genes")
    ax.scatter(g.cdsLength[dup], g.densMAP_ro[dup], s=12, facecolor="none", edgecolor=COPIES, lw=0.7,
               marker="^", label=f"copias idénticas: {dup.sum()}")
    ax.axvline(1000, color="0.4", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Longitud del CDS en nucleótidos")
    ax.set_ylabel("Log-radio local original, log $\\mathbf{r_o}$")
    ax.text(0.98, 0.03, "eje x en escala log", transform=ax.transAxes, ha="right", va="bottom", **NOTE)
    short, long_ = g.cdsLength < 1000, g.cdsLength >= 1000
    rs = spearmanr(g.cdsLength[short], g.densMAP_ro[short])[0]
    rl = spearmanr(g.cdsLength[long_], g.densMAP_ro[long_])[0]
    ax.text(900, 0.97, f"< 1000 nt\nρ = {rs:.2f}".replace("-", "−"), transform=ax.get_xaxis_transform(),
            ha="right", va="top")
    ax.text(1100, 0.97, f"≥ 1000 nt\nρ = {rl:.2f}".replace("-", "−"), transform=ax.get_xaxis_transform(),
            ha="left", va="top")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, markerscale=1.5,
              handletextpad=0.2, borderaxespad=0.1)
    Title(ax, "b", "Radio original contra longitud del CDS", pad=16)
    fig.tight_layout(h_pad=1.0)
    Save(fig, "falla_escenarios_longitud.pdf", dpi=600)
    plt.close(fig)
    print(f"Falla: Spearman ro-CDS <1000 = {rs:.3f} (n={short.sum()}), >=1000 = {rl:.3f} (n={long_.sum()}),"
          f" total = {spearmanr(g.cdsLength, g.densMAP_ro)[0]:.3f}; copias = {dup.sum()} en"
          f" {g[dup].groupby(['PCA_x', 'PCA_y']).ngroups} grupos; mediana ro copias = {g.densMAP_ro[dup].median():.2f},"
          f" total = {g.densMAP_ro.median():.2f}")


if __name__ == "__main__":
    Sintetico()
    Sensibilidad()
    Reales()
    Falla()
