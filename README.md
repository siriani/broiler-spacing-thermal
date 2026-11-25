# Camera-measured spacing of broilers under cold, thermoneutral and heat treatments: code and data

Code and data for the article *"Camera-Measured Spacing of Broiler Chickens Under Cold, Thermoneutral and Heat Treatments"* (submitted to *Animals*, MDPI).

**Repository authors:** Igor Cristian Hübner and Allan Lincoln Rodrigues Siriani.

**Archived version:** [doi:10.5281/zenodo.24518994](https://doi.org/10.5281/zenodo.24518994)

## What is here

| Path | Content |
|---|---|
| `dados/medicao_v0/` | Per-bird detections of the segmentation model v1, one frame every 15 s, for the seven recording days. Each row gives position, projected area and ellipse axes in cm on the litter plane. The files are before the CAM2 recalibration and are the input of `corrige_cam2.py` |
| `dados/calibracao_v1.json`, `cantos_cam2_por_lote.json`, `distorcao.json` | Camera homographies, CAM2 litter corners per period (CAM2 was reframed between and within batches) and lens-distortion coefficients |
| `dados/fundos_equip/`, `dados/equipamentos_por_data.csv` | Median background frame per date and camera, and the feeder and drinker positions measured on them (px and cm) |
| `dados/registrador/` | Air temperature and relative humidity from the data logger of each compartment (Elitech RCW-800W V2, 0.20 m above the floor, 5-min interval). Only the seven recording days are included, from 1 h before the first to 1 h after the last analysed frame |
| `dados/Table_S4_weekly_weights.csv`, `pesagens_peso_medio_por_ave.csv` | Weekly group weights (mean per bird) |
| `dados/quadros_figura3/` | The three video frames shown in Figure 3 and how they were selected |
| `analise/`, `figuras/` | Analysis and figure scripts |

**Videos.** Eight of the recordings, with manual annotations and tracking outputs, are openly available in the dataset of the companion study: Hübner, Siriani and Moura, *Overhead videos, manual annotations and per-instance tracking outputs of small broiler groups*, Zenodo, [doi:10.5281/zenodo.23960761](https://doi.org/10.5281/zenodo.23960761). The remaining recordings are available from the authors on reasonable request because of their size. All analyses in this repository start from the per-bird detections in `dados/medicao_v0/`, so the videos are not needed to reproduce the results.

## How to reproduce

```bash
pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` runs all the steps and writes the outputs to `analise/`, `figuras/` and `suplemento/`:
- recalibrates CAM2;
- computes the per-frame and per-compartment-date variables;
- runs the randomisation tests and sensitivity analyses (`analise/stats_confirmatoria.txt`, `sensibilidade*.txt`);
- computes the logger temperature during the analysed frames (Table 2, Table S3);
- computes the drinker and feeder zones (Table S6, `analise/zonas_teste.txt`) and Text S2;
- draws Figures 1–3.

Tested with Python 3.13; the outputs were identical to those reported in the article.

## Design

- **Experiment:** three 42-day batches (L1–L3) of ten Ross broilers per compartment, three compartments per batch, in the FEAGRI/UNICAMP climate chamber. The animal study was approved by CEUA/UNICAMP (protocol 6519-1/2024).
- **Treatments:** cold, thermoneutral and heat were randomised to the compartments within each batch and applied in 8-h sessions (08:00–16:00) every 3–4 days.
- **Variable names:** the scripts use Portuguese names.

  | Portuguese | English |
  |---|---|
  | `frio` | cold |
  | `termoneutro` | thermoneutral |
  | `calor` | heat |
  | `lote` | batch |
  | `box` | compartment |
  | `idade` | age (d) |
  | `bebedouro` | drinker |
  | `comedouro` | feeder |

## Licence

- **Code:** MIT (`LICENSE`).
- **Data:** CC BY 4.0.

## Citation

See `CITATION.cff`.
