"""Validation of the lens-distortion correction for distances (Article 2).
(1) Map, for each camera, the ratio corrected/uncorrected of a 25-cm distance at each litter position.
(2) Effect of the correction on the per-box-date median NND (uncorrected vs corrected), by treatment.
Outputs: figuras/distorcao_fator_distancia.png, analise/distorcao_efeito_nnd.csv"""
import json, numpy as np, pandas as pd, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import importlib.util
spec=importlib.util.spec_from_file_location("e","analise/explora_espacamento.py")
src=open("analise/explora_espacamento.py").read(); corretor_src=src[:src.index("rows=[]")]
ns={}; exec(corretor_src,ns); corretor=ns["corretor"]; CAL=ns["CAL"]
fig,axs=plt.subplots(1,3,figsize=(12,3.6))
for ax,cam in zip(axs,(1,2,3)):
    L,A=CAL[f"CAM{cam}"]["dim_cm"]; f=corretor(cam)
    xs=np.linspace(15,L-15,40); ys=np.linspace(15,A-15,34); R=np.zeros((len(ys),len(xs)))
    for i,y in enumerate(ys):
        for j,x in enumerate(xs):
            P=np.array([[x-12.5,y],[x+12.5,y],[x,y-12.5],[x,y+12.5]])
            Q=f(P); R[i,j]=(np.linalg.norm(Q[1]-Q[0])+np.linalg.norm(Q[3]-Q[2]))/50
    im=ax.imshow(R,extent=[xs[0],xs[-1],ys[-1],ys[0]],cmap="RdBu_r",vmin=0.9,vmax=1.1)
    ax.set_title(f"CAM{cam}"+(" (not corrected: k not identifiable)" if cam==2 else f"  range {R.min():.3f} to {R.max():.3f}"),fontsize=9)
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
fig.colorbar(im,ax=axs,shrink=0.85,label="corrected / uncorrected distance")
plt.savefig("figuras/distorcao_fator_distancia.png",dpi=250,bbox_inches="tight")
# (2) effect on NND
import glob
from scipy.spatial import cKDTree
rows=[]
for fcsv in sorted(glob.glob("dados/medicao_v0/seg_todas_idades_v1_best/rep*.csv")):
    d=pd.read_csv(fcsv); 
    if d.empty: continue
    cam=int(d.box.iloc[0]); f=corretor(cam)
    for q,g in d[d.n_aves_quadro==10].groupby("quadro"):
        P=g[["x_cm","y_cm"]].to_numpy(float); Q=f(P)
        a=cKDTree(P).query(P,k=2)[0][:,1].mean(); b=cKDTree(Q).query(Q,k=2)[0][:,1].mean()
        rows.append(dict(lote=g.lote.iloc[0],idade=int(g.idade_d.iloc[0]),box=cam,trat=g.tratamento.iloc[0],nnd_sem=a,nnd_com=b))
F=pd.DataFrame(rows); E=F.groupby(["lote","idade","box","trat"])[["nnd_sem","nnd_com"]].median().reset_index()
E["razao"]=E.nnd_com/E.nnd_sem; E.to_csv("analise/distorcao_efeito_nnd.csv",index=False)
print(E.groupby("box").razao.agg(["min","median","max"]).round(4)); print(E.groupby("trat").razao.agg(["min","median","max"]).round(4))
