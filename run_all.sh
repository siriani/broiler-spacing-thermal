#!/usr/bin/env bash
# Reproduces every table, test and figure of the article from the per-bird detections (run from the repository root).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p suplemento video
python analise/corrige_cam2.py            # CAM2 recalibration per period -> dados/medicao_v1
python analise/explora_espacamento.py     # per-frame and per-compartment-date variables
python analise/equipamentos.py            # feeder and drinker positions (px -> cm)
python analise/padroes.py                 # hourly comparison; Figure 2
python analise/stats_confirmatoria.py     # randomisation tests, mixed model (Section 3.2)
python analise/temperatura_janelas.py     # logger temperature during the analysed frames (Table 2, Table S3)
python analise/sensibilidade.py           # frame retention (Table S2) and sensitivity analyses (Table S5)
python analise/sensibilidade2.py          # leave-one-date-out, compartment effects, confidence sets
python figuras/fig_zonas.py               # drinker and feeder zones per compartment-date (Table S6)
python analise/zonas_teste.py             # tests of the drinker and feeder zones (Section 3.3)
python analise/texto_s2.py                # within-compartment relation with logged temperature (Text S2)
python figuras/fig_setup.py               # Figure 1
python figuras/fig_recursos.py            # Figure 3
python analise/checagens_r11.py         # per-batch differences, joint interval, random-placement NND, corrected NND_BL, 8–12 detections
python analise/exporta_suplemento.py     # Tables S2, S3 and S6 in English, as published
