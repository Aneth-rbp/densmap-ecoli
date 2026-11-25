# densMAP en embeddings de Evo 2 de genes de *E. coli*

Código y resultados del artículo "densMAP frente a UMAP y PCA" (Matemáticas, Proyecto U2,
Maestría, CUCEI, Universidad de Guadalajara). El artículo compara densMAP, UMAP y PCA en tres
datos:

- un sintético de tres gaussianas con σ 1:2:4 en 20 dimensiones;
- Breast Cancer Wisconsin, que viene incluido en scikit-learn;
- los embeddings de Evo 2 7B de 3583 genes de *E. coli* K-12 MG1655.

## Contenido

| Ruta | Qué es |
|---|---|
| `DensmapExperiments.py` | Todos los experimentos: datos de prueba de los autores, sintético, barridos de λ y k, escenarios, benchmark y genes. |
| `RunAuthorsScript.py` | Corre sin cambios el script de prueba de los autores (`trial_densmap.py`) en numba 0.50 (aarch64); su salida está en `resultados_autores/trial/`. Solo cambia el import de `numba.errors` y compila `optimize_layout` sin `parallel`; el algoritmo no cambia. `DensmapExperiments.py` aplica los mismos dos ajustes con `--backend authors`. |
| `PlotSideBySide.py` | UMAP y densMAP lado a lado, con la misma escala de ejes. |
| `figuras/HacerFiguras.py` | Figuras del artículo, leídas de `resultados_autores/` (no recalcula embeddings). |
| `figuras/` | Las figuras tal como aparecen en el artículo. |
| `resultados_autores/` | CSV de métricas (media y desviación estándar de 5 semillas, y por semilla en `*_raw.csv`), coordenadas y radios locales por gen (`real_genes_seed0.csv`), figuras de diagnóstico y bitácoras. |
| `pilotos/` | Pruebas iniciales con `umap-learn`; no se usan en el artículo. |

## Referencias de código

- **densMAP original** (Narayan, Berger y Cho, 2021): https://github.com/hhcho/densvis,
  paquete `densmap-learn` 0.2.1, commit `d65bb31`.
- **Embeddings de los genes:** salen de https://github.com/Aneth-rbp/sae-ecoli-dna, capa
  `blocks.28.mlp.l3` de Evo 2 7B, promediada sobre los primeros 1000 nt de cada CDS. Ese repo
  también descarga el genoma, la anotación GO y define los genes. Los embeddings (`.npy`) no
  están en ningún repo porque se generan con GPU.

## Entorno de la corrida

Las corridas se hicieron en una NVIDIA DGX Spark (aarch64), en un entorno conda con:

- Python 3.8.19
- `densmap-learn` 0.2.1, instalado desde densvis
- numba 0.50.1 y llvmlite 0.33.0
- scikit-learn 0.24.2
- numpy 1.19.5
- scipy 1.9.1
- pandas 1.4.4

## Reproducir

```bash
git clone https://github.com/hhcho/densvis.git
pip install -e densvis/densmap

# Datos de prueba de los autores, sintético, barridos y benchmark (5 semillas)
python DensmapExperiments.py resultados_autores \
    densvis/densmap/densmap/trial_data.txt --backend authors

# Genes de E. coli: necesita data/ de sae-ecoli-dna, con embeddings y Genes.tsv
python DensmapExperiments.py resultados_autores \
    densvis/densmap/densmap/trial_data.txt --backend authors \
    --steps real --real-data ../sae-ecoli-dna/data

# Figuras del artículo
python figuras/HacerFiguras.py
```
