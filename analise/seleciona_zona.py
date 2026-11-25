"""Escolhe, para 20 d (L1), entre 11 h e 13 h, o quadro de cada tratamento cuja % de aves na zona do bebedouro (<= 30 cm)
é a mais próxima da média do compartimento no dia. Saída: dados/quadros_figura3/selecao_zona.csv"""
import glob,re,numpy as np,pandas as pd
from datetime import datetime,timedelta
src=open("analise/explora_espacamento.py").read(); ns={}; exec(src[:src.index("rows=[]")],ns); corretor=ns["corretor"]
E=pd.read_csv("dados/equipamentos_por_data.csv"); out=[]
for cam in [1,2,3]:
  rows=[]
  for f in glob.glob(f"dados/medicao_v1/seg_todas_idades_v1_best/rep1_2026-02-24_CAM{cam}*.csv"):
    d=pd.read_csv(f); d=d[d.n_aves_quadro==10]
    if d.empty: continue
    v=d.video.iloc[0]; m=re.search(r"(\d{4})-?(\d{2})-?(\d{2})-?(\d{2})-?(\d{2})-?(\d{2})",v); t0=datetime(*map(int,m.groups()))
    Q=E[(E.lote=="rep1")&(E.data=="2026-02-24")&(E.box==cam)&(E.tipo=="bebedouro")][["x_cm","y_cm"]].to_numpy(float)
    for q,g in d.groupby("quadro"):
        P=corretor(cam)(g[["x_cm","y_cm"]].to_numpy(float)); z=(np.linalg.norm(P[:,None]-Q[None],axis=2).min(1)<=30).mean()
        t=t0+timedelta(seconds=float(g.t_s.iloc[0])); rows.append(dict(video=v,trat=d.tratamento.iloc[0],quadro=q,t_s=float(g.t_s.iloc[0]),h=t.hour+t.minute/60,hora=t.strftime("%H:%M"),z=z))
  R=pd.DataFrame(rows); alvo=R[R.h<16].z.mean(); w=R[(R.h>=10)&(R.h<12)]; r=w.iloc[(w.z-alvo).abs().argmin()]
  out.append(dict(lote="rep1",data="2026-02-24",idade=20,trat=r.trat,box=cam,video=r.video,quadro=r.quadro,t_s=r.t_s,hora=r.hora,z=r.z,z_dia=alvo,tipo="zona"))
S=pd.DataFrame(out); S.to_csv("dados/quadros_figura3/selecao_zona.csv",index=False); print(S[["trat","hora","z","z_dia"]].round(2))
