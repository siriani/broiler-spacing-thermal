"""Corrige as medições da CAM2 nos lotes 1 e 2 (câmera reenquadrada; a calibracao_v1 só vale no lote 3).
cm (homografia antiga) -> px -> cm (homografia nova, cantos em dados/cantos_cam2_por_lote.json).
Área e eixos: escala local da transformação composta (área × det J; eixos × sqrt(det J)).
Também corrige 18/06 (L3), quando a CAM2 foi reenquadrada de novo (chave rep3_2026-06-18).
Entrada: dados/medicao_v0/...; saída: dados/medicao_v1/... (as outras câmeras e o lote 3 são copiados sem mudança)."""
import glob, json, os, shutil
import numpy as np, pandas as pd, cv2
A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL = json.load(open(f"{A}/dados/calibracao_v1.json"))["CAM2"]
K = json.load(open(f"{A}/dados/cantos_cam2_por_lote.json"))
L, W = CAL["dim_cm"]; dst = np.float32([[0, 0], [L, 0], [L, W], [0, W]])
H_old = np.array(CAL["H_img_para_cm"], float)
chk = cv2.getPerspectiveTransform(np.float32(K["rep3"]), dst)
assert np.allclose(chk / chk[2, 2], H_old / H_old[2, 2], atol=1e-6), "convenção da homografia difere"
def T(chave):
    Hn = cv2.getPerspectiveTransform(np.float32(K[chave]), dst).astype(float)
    return Hn @ np.linalg.inv(H_old)
def ap(M, P):
    q = np.c_[P, np.ones(len(P))] @ M.T; return q[:, :2] / q[:, 2:]
src = f"{A}/dados/medicao_v0"; out = f"{A}/dados/medicao_v1"
if os.path.exists(out): shutil.rmtree(out)
shutil.copytree(src, out)
log = []
for f in sorted(glob.glob(f"{out}/seg_todas_idades_v1_best/rep*.csv")):
    d = pd.read_csv(f)
    if d.empty or int(d.box.iloc[0]) != 2: continue
    chave = f"{d.lote.iloc[0]}_{d.data.iloc[0]}" if f"{d.lote.iloc[0]}_{d.data.iloc[0]}" in K else d.lote.iloc[0]
    if chave == "rep3": continue   # calibração v1 marcada no L3 (vale para 21/05); 18/06 tem cantos próprios (reenquadrada)
    M = T(chave); P = d[["x_cm", "y_cm"]].to_numpy(float); e = 1e-3
    Pn = ap(M, P); Jx = (ap(M, P + [e, 0]) - Pn) / e; Jy = (ap(M, P + [0, e]) - Pn) / e
    det = np.abs(Jx[:, 0] * Jy[:, 1] - Jx[:, 1] * Jy[:, 0])
    d["x_cm"], d["y_cm"] = Pn[:, 0], Pn[:, 1]
    d["area_cm2"] *= det; d["eixo_maior_cm"] *= np.sqrt(det); d["eixo_menor_cm"] *= np.sqrt(det)
    d.to_csv(f, index=False)
    log.append(f"{os.path.basename(f)}: escala de área média {det.mean():.3f}; deslocamento médio {np.linalg.norm(Pn - P, axis=1).mean():.1f} cm")
print("\n".join(log))
