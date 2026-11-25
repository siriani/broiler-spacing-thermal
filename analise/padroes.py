"""Pattern search by thermal treatment (Article 2). Exact-10 frames, lens-corrected positions (CAM1, CAM3; CAM2 uncorrected).
Main spacing metric: NND in body lengths (distortion-invariant; see CORRECAO_DISTORCAO.md)."""
import glob, re, json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from datetime import datetime, timedelta
src=open("analise/explora_espacamento.py").read(); ns={}; exec(src[:src.index("rows=[]")],ns); corretor=ns["corretor"]; CAL=ns["CAL"]
COR={"frio":"#2b6cb0","termoneutro":"#2f855a","calor":"#c53030"}; ORD=["frio","termoneutro","calor"]
fr=[]; birds=[]
for f in sorted(glob.glob("dados/medicao_v1/seg_todas_idades_v1_best/rep*.csv")):
    d=pd.read_csv(f)
    if d.empty: continue
    v=d.video.iloc[0]; m=re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})",v) or re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})",v)
    t0=datetime(*map(int,m.groups())); cam=int(d.box.iloc[0]); corr=corretor(cam); L,A=CAL[f"CAM{cam}"]["dim_cm"]
    for q,g in d[d.n_aves_quadro==10].groupby("quadro"):
        P0=g[["x_cm","y_cm"]].to_numpy(float); P=corr(P0); ax=g.eixo_maior_cm.to_numpy(float); mn=g.eixo_menor_cm.to_numpy(float)
        dd,ii=cKDTree(P0).query(P0,k=2); nbl=dd[:,1]/ax
        # groups: link birds closer than 1.5 mean body lengths (uncorrected, local scale)
        D=np.linalg.norm(P0[:,None]-P0[None],axis=2); adj=D<1.5*ax.mean()
        ncomp,lab=connected_components(adj,directed=False); big=np.bincount(lab).max()
        t=t0+timedelta(seconds=float(g.t_s.iloc[0]))
        fr.append(dict(lote=g.lote.iloc[0],idade=int(g.idade_d.iloc[0]),data=g.data.iloc[0],box=cam,trat=g.tratamento.iloc[0],hora=t.hour+t.minute/60,
            nnd_bl=nbl.mean(),ngrupos=ncomp,maior_grupo=big,ar=(ax/mn).mean(),area=g.area_cm2.mean()))
        for (x,y),a,r in zip(P,g.area_cm2,ax/mn): birds.append((g.lote.iloc[0],int(g.idade_d.iloc[0]),cam,g.tratamento.iloc[0],x/L,y/A,a,r))
F=pd.DataFrame(fr); Bd=pd.DataFrame(birds,columns=["lote","idade","box","trat","xn","yn","area","ar"])
F.to_csv("analise/padroes_quadros.csv",index=False)
out=[]
# 1) time-matched comparison: hours common to the 3 pens of each date
F["h"]=F.hora.astype(int); res=[]
for data,g in F.groupby("data"):
    hs=set.intersection(*[set(x.h) for _,x in g.groupby("trat")])
    for h in sorted(hs):
        s=g[g.h==h].groupby("trat").nnd_bl.median()
        if len(s)==3: res.append(dict(data=data,h=h,frio_menor=s.idxmin()=="frio",calor_maior=s.idxmax()=="calor"))
R=pd.DataFrame(res); out.append(f"Janelas de 1 h comuns aos 3 boxes: {len(R)} (em {R.data.nunique()} datas). NND/corpo: frio menor em {R.frio_menor.sum()}/{len(R)}; calor maior em {R.calor_maior.sum()}/{len(R)}.")
# 2) per box-date medians of other patterns
M=F.groupby(["lote","idade","data","box","trat"])[["nnd_bl","ngrupos","maior_grupo","ar","area"]].median().reset_index()
M.to_csv("analise/padroes_box_data.csv",index=False)
for col,lab in [("ngrupos","n. de grupos"),("maior_grupo","maior grupo"),("ar","razao eixos (alongamento)")]:
    rk=M.groupby("data")[["trat",col]].apply(lambda g: g.set_index("trat")[col].rank(method="min")); rk=rk.unstack() if isinstance(rk,pd.Series) else rk
    out.append(f"{lab}: mediana por tratamento "+", ".join(f"{t} {M[M.trat==t][col].median():.2f}" for t in ORD))
# figure A: NND/body by age
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
mk={"rep1":"o","rep2":"s","rep3":"^"}
for t in ORD:
    s=M[M.trat==t]
    for _,r in s.iterrows(): ax[0].plot(r.idade+{"frio":-0.6,"termoneutro":0,"calor":0.6}[t],r.nnd_bl,mk[r.lote],color=COR[t],ms=5)
    ax[0].plot([],[],"o",color=COR[t],label={"frio":"Cold","termoneutro":"Thermoneutral","calor":"Heat"}[t])
for l,m in mk.items(): ax[0].plot([],[],m,color="k",mfc="none",label=l.replace("rep","Batch L"))
ax[0].set_xlabel("Age (d)"); ax[0].set_ylabel("Nearest-neighbour distance (body lengths)"); ax[0].legend(frameon=False,fontsize=8,ncol=2); ax[0].set_title("(a)",loc="left",fontsize=10,fontweight="bold")
# figure A right: diurnal course for long recordings
for data,g in F.groupby("data"):
    if g.h.nunique()<4: continue
    for t in ORD:
        s=g[g.trat==t].groupby("h").nnd_bl.median(); ax[1].plot(s.index,s.values,"-o",ms=3,color=COR[t],alpha=0.8)
    ax[1].text(s.index[-1]+0.1,s.values[-1],f"{g.idade.iloc[0]} d",fontsize=7)
ax[1].axvspan(16,17.4,color="0.9",zorder=0); ax[1].text(16.7,ax[1].get_ylim()[1]*0.985,"after\nsession",fontsize=7,ha="center",va="top",color="0.4")
ax[1].set_xlabel("Clock hour"); ax[1].set_ylabel("NND (body lengths)"); ax[1].set_title("(b)",loc="left",fontsize=10,fontweight="bold"); ax[1].set_title("Long recordings: 8 d (L3), 20 and 40 d (L1)",fontsize=9)
plt.tight_layout(); plt.savefig("figuras/nnd_idade_hora.png",dpi=250)
# figure B: floor-use maps per date x treatment (corrected positions; CAM2 uncorrected)
EQ=pd.read_csv("dados/equipamentos_por_data.csv")
dates=sorted(M.data.unique(),key=lambda x:(M[M.data==x].idade.iloc[0],x))
fig,axs=plt.subplots(3,len(dates),figsize=(2.0*len(dates),5.6))
for j,data in enumerate(dates):
    lote=M[M.data==data].lote.iloc[0]; idade=M[M.data==data].idade.iloc[0]
    for i,t in enumerate(ORD):
        b=Bd[(Bd.lote==lote)&(Bd.idade==idade)&(Bd.trat==t)]; cam=M[(M.data==data)&(M.trat==t)].box.iloc[0]
        h,_,_=np.histogram2d(b.yn.clip(0,1),b.xn.clip(0,1),bins=[13,16],range=[[0,1],[0,1]])
        axs[i,j].imshow(h/h.sum(),cmap="magma",vmin=0,vmax=0.03); axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
        for _,q in EQ[(EQ.lote==lote)&(EQ.data==data)&(EQ.box==cam)].iterrows():
            axs[i,j].plot(q.xn*16-0.5,q.yn*13-0.5,"s" if q.tipo=="comedouro" else "o",mfc="none",mec="cyan" if q.tipo=="comedouro" else "white",ms=7 if q.tipo=="comedouro" else 5.5,mew=1.2)
        axs[i,j].set_title(f"{idade} d {lote.replace('rep','L')} C{cam}",fontsize=7) if i==0 else axs[i,j].set_title(f"C{cam}",fontsize=7)
    axs[0,0].set_ylabel("Cold",fontsize=8); axs[1,0].set_ylabel("Thermoneutral",fontsize=8); axs[2,0].set_ylabel("Heat",fontsize=8)
plt.suptitle("Where birds stand: share of bird positions per floor cell (exact-10 frames; square = feeder, circles = drinkers)",fontsize=9); plt.tight_layout(); plt.savefig("figuras/mapas_ocupacao.png",dpi=200)
print("\n".join(out)); print(M.sort_values(["idade","lote","trat"]).round(3).to_string(index=False))
