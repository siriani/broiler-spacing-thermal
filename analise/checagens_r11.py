"""Rodada 11 (revisão Fable): (a) conjunto de confiança conjunto (δc, δh) por inversão do teste de aleatorização, projetado;
(b) diferenças médias ponderadas por lote; (c) NND sob posicionamento aleatório (Monte Carlo 2000 por câmera, com distância mínima de 0,6 comprimento corporal
e sem exclusão; ignora paredes, comedouro e bebedouros); (d) NND_BL com posições e escala corrigidas de distorção (CAM1/CAM3); (e) NND_BL com todos os quadros de 8–12 detecções.
Saída: analise/checagens_r11.txt"""
import glob,itertools,numpy as np,pandas as pd
from scipy.spatial import cKDTree
out=[];P=lambda *a: out.append(" ".join(str(x) for x in a)); rng=np.random.default_rng(1)
B=pd.read_csv("analise/box_data.csv"); T3=["frio","termoneutro","calor"]; lotes=sorted(B.lote.unique())
# (a) joint inversion
dates=sorted(B.data.unique()); D=len(dates)
Yv=np.array([[B[(B.data==d)&(B.box==b)].nnd_bl.iloc[0] for b in (1,2,3)] for d in dates])
lot=[B[B.data==d].lote.iloc[0] for d in dates]
obs={(l,b):B[(B.lote==l)&(B.box==b)].trat.iloc[0] for l in lotes for b in (1,2,3)}
code={"frio":0,"termoneutro":1,"calor":2}
A_obs=np.array([[code[obs[(lot[i],b)]] for b in (1,2,3)] for i in range(D)])
allocs=[]
for perms in itertools.product(list(itertools.permutations([0,1,2])),repeat=3):
    m={l:p for l,p in zip(lotes,perms)}; allocs.append(np.array([[m[lot[i]][b] for b in range(3)] for i in range(D)]))
allocs=np.array(allocs)  # 216 x D x 3
def diffs(Y,A):  # mean over dates of cold−TN, heat−TN; Y: D x3, A: (...,D,3)
    pick=lambda k: (Y*(A==k)).sum(-1)
    return (pick(0)-pick(1)).mean(-1),(pick(2)-pick(1)).mean(-1)
oc,oh=diffs(Yv,A_obs); P(f"(a) observado frio−TN {oc:+.4f}, calor−TN {oh:+.4f} (média de datas)")
grid=np.round(np.arange(-0.25,0.2501,0.005),3); acc=[]
for dc in grid:
    for dh in grid:
        Y0=Yv-dc*(A_obs==0)-dh*(A_obs==2)
        c,h=diffs(Y0,allocs); c0,h0=diffs(Y0,A_obs)
        s=c**2+h**2; s0=c0**2+h0**2
        if (s>=s0-1e-12).mean()>0.10: acc.append((dc,dh))
acc=np.array(acc)
P(f"    região conjunta 90% projetada: frio−TN {acc[:,0].min():+.3f} a {acc[:,0].max():+.3f}; calor−TN {acc[:,1].min():+.3f} a {acc[:,1].max():+.3f} (grade 0,005; limite da grade ±0,25)")
# (b) batch-weighted
W=B.pivot_table(index=["lote","data"],columns="trat",values="nnd_bl").reset_index()
L=W.assign(c=W.frio-W.termoneutro,h=W.calor-W.termoneutro).groupby("lote")[["c","h"]].mean()
P(f"(b) por lote: frio−TN {L.c.round(3).tolist()} média {L.c.mean():+.3f}; calor−TN {L.h.round(3).tolist()} média {L.h.mean():+.3f}")
# per-bird data
src=open("analise/explora_espacamento.py").read(); ns={}; exec(src[:src.index("rows=[]")],ns); corretor=ns["corretor"]; CAL=ns["CAL"]
rows=[];allf=[]
for f in glob.glob("dados/medicao_v1/seg_todas_idades_v1_best/rep*.csv"):
    d=pd.read_csv(f)
    if d.empty: continue
    cam=int(d.box.iloc[0]); cf=corretor(cam)
    for q,g in d.groupby("quadro"):
        n=len(g)
        if n<8 or n>12 or n<2: continue
        Pp=g[["x_cm","y_cm"]].to_numpy(float); ax=g.eixo_maior_cm.to_numpy(float)
        dd,_=cKDTree(Pp).query(Pp,k=2); nb=(dd[:,1]/ax).mean()
        r=dict(lote=d.lote.iloc[0],data=d.data.iloc[0],box=cam,trat=d.tratamento.iloc[0],n=n,nnd_bl=nb,nnd=dd[:,1].mean(),ax=ax.mean())
        if n==10:
            Pc=cf(Pp); sx=np.linalg.norm(cf(Pp+[1,0])-Pc,axis=1); sy=np.linalg.norm(cf(Pp+[0,1])-Pc,axis=1)
            dc_,_=cKDTree(Pc).query(Pc,k=2); r["nnd_bl_corr"]=(dc_[:,1]/(ax*(sx+sy)/2)).mean()
        rows.append(r)
F=pd.DataFrame(rows)
def batch_test(V,col):
    M=V.groupby(["lote","data","trat"])[col].median().reset_index().pivot_table(index=["lote","data"],columns="trat",values=col)
    Lm=M.groupby("lote").mean(); c=(Lm.idxmin(axis=1)=="frio").sum(); h=(Lm.idxmax(axis=1)=="calor").sum()
    batch_test.Lm=Lm
    return c,h,(M.frio/M.termoneutro).mean(),(M.calor/M.termoneutro).mean()
for lab,V,col in [("(d) NND_BL corrigida de distorção (n=10)",F[F.n==10],"nnd_bl_corr"),("    NND_BL original (n=10)",F[F.n==10],"nnd_bl"),("(e) NND_BL com quadros de 8–12 detecções",F,"nnd_bl")]:
    c,h,rc,rh=batch_test(V,col); P(f"{lab}: frio menor {c}/3, calor maior {h}/3; frio/TN {rc:.3f}, calor/TN {rh:.3f}"); P("      médias por lote:",batch_test.Lm[T3].round(3).to_dict('index'))
# (c) random NND reference, arena per camera from calibration
P("(c) NND (cm) sob posicionamento aleatório de 10 aves (Monte Carlo 2000):")
T10=F[F.n==10]
for cam in (1,2,3):
    Lc,Ac=CAL[f"CAM{cam}"]["dim_cm"]; obsn=T10[T10.box==cam].groupby("data").nnd.median().mean(); axm=T10[T10.box==cam].ax.mean()
    def sim(excl):
        v=[]
        for _ in range(2000):
            pts=[]
            while len(pts)<10:
                p=rng.uniform([0,0],[Lc,Ac])
                if all(np.hypot(*(p-q))>=excl for q in pts): pts.append(p)
            pts=np.array(pts); dd,_=cKDTree(pts).query(pts,k=2); v.append(dd[:,1].mean())
        return np.mean(v)
    r0=sim(0); r1=sim(axm*0.6)
    P(f"   CAM{cam} {Lc}×{Ac} cm: aleatório {r0:.1f} cm; com exclusão de 0,6 comprimento ({axm*0.6:.1f} cm) {r1:.1f} cm; observado (média das datas) {obsn:.1f} cm")
G=T10.groupby(["data","trat"]).nnd.median().groupby("trat").mean(); P("   observado por tratamento (cm):",G.round(1).to_dict())
open("analise/checagens_r11.txt","w").write("\n".join(out)); print("\n".join(out))
