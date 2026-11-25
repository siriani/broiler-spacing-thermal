"""Artigo 2, exploratório: espaçamento e postura das aves por tratamento térmico (frio, termoneutro, calor).
Entrada: medições por ave do Artigo 1 (modelo v1, 1 quadro a cada 15 s, conf >= 0,25; dados/medicao_v1).
Unidade: box-data (lote x idade x box). Usa só quadros com exatamente 10 aves (contagem conhecida)."""
import glob, re, json, numpy as np, pandas as pd
from scipy.spatial import cKDTree, ConvexHull
from datetime import datetime, timedelta
CAL=json.load(open("dados/calibracao_v1.json")); DIST=json.load(open("dados/distorcao.json"))
import cv2
def corretor(cam):
    """cm (homography from distorted px) -> cm after one-parameter division-model undistortion (Article 1, Suppl. Note 1). CAM2 left uncorrected."""
    C=CAL[f"CAM{cam}"]; H=np.array(C["H_img_para_cm"]); Hi=np.linalg.inv(H); W,Hh=C["res"]; c=np.array([W/2,Hh/2])
    k=DIST.get(f"CAM{cam}",{}).get("k") if cam!=2 else None
    if not k: return lambda P: P
    und=lambda p: c+(p-c)/(1+k*((p-c)**2).sum(1,keepdims=True))
    cpx=np.array(C["cantos_px"],float); L,A=C["dim_cm"]
    dst=np.array([[0,0],[L,0],[L,A],[0,A]],float)
    Hu=cv2.findHomography(und(cpx),dst)[0]; H0=cv2.findHomography(cpx,dst)[0]
    def f(P):
        px=cv2.perspectiveTransform(P.reshape(-1,1,2).astype(np.float64),np.linalg.inv(H0)).reshape(-1,2)
        return cv2.perspectiveTransform(und(px).reshape(-1,1,2),Hu).reshape(-1,2)
    return f
rows=[]
for f in sorted(glob.glob("dados/medicao_v1/seg_todas_idades_v1_best/rep*.csv")):
    d=pd.read_csv(f)
    if d.empty: continue
    v=d.video.iloc[0]; data=d.data.iloc[0]
    m=re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})",v) or re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})",v)
    t0=datetime(*map(int,m.groups())) if m else None
    cam=int(d.box.iloc[0]); L,A=CAL[f"CAM{cam}"]["dim_cm"]; corr=corretor(cam)
    for q,g in d[d.n_aves_quadro==10].groupby("quadro"):
        P0=g[["x_cm","y_cm"]].to_numpy(float); P=corr(P0)
        nnd0=cKDTree(P0).query(P0,k=2)[0][:,1]
        dd,_=cKDTree(P).query(P,k=2); nnd=dd[:,1]
        ax=g.eixo_maior_cm.to_numpy(float)
        cen=P.mean(0); disp=np.linalg.norm(P-cen,axis=1).mean()
        wall=np.minimum.reduce([P[:,0],P[:,1],L-P[:,0],A-P[:,1]])
        try: hull=ConvexHull(P).volume
        except Exception: hull=np.nan
        t=t0+timedelta(seconds=float(g.t_s.iloc[0])) if t0 else None
        rows.append(dict(lote=g.lote.iloc[0],idade=int(g.idade_d.iloc[0]),box=cam,trat=g.tratamento.iloc[0],data=data,
            hora=(t.hour+t.minute/60) if t else np.nan,nnd=nnd.mean(),nnd_sem=nnd0.mean(),nnd_bl=(nnd0/ax).mean(),area=g.area_cm2.mean(),
            ocup=g.area_cm2.sum()/(L*A),disp=disp,parede=(wall<10).mean(),hull=hull/(L*A)))
F=pd.DataFrame(rows); F.to_csv("analise/quadros_10aves.csv",index=False)
M=["nnd","nnd_sem","nnd_bl","area","ocup","disp","parede","hull"]
B=F.groupby(["lote","idade","data","box","trat"])[M+["hora"]].median().reset_index()
B["n_quadros"]=F.groupby(["lote","idade","data","box","trat"]).size().values
B.to_csv("analise/box_data.csv",index=False)
pd.set_option("display.width",200)
print(B.sort_values(["idade","lote","trat"]).round(3).to_string(index=False))
print("\nOrdem dos tratamentos em cada data (1 = menor):")
for m in M:
    r=B.groupby("data")[["trat",m]].apply(lambda g: g.set_index("trat")[m].rank().to_dict())
    tab=pd.DataFrame(list(r.values),index=r.index)
    print(f"{m:7s} frio=1 em {int((tab['frio']==1).sum())}/{len(tab)} datas; calor=3 em {int((tab['calor']==3).sum())}/{len(tab)}; ranks frio {tab['frio'].tolist()} calor {tab['calor'].tolist()}")
