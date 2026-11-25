"""Text S2: Spearman entre NND_BL horária e temperatura média horária do registrador do próprio compartimento,
nas datas com >= 4 h (8 d L3; 20 e 40 d L1), com e sem as horas a partir das 16 h. Saída: analise/texto_s2.txt"""
import glob,pandas as pd
from scipy.stats import spearmanr
import sys; P=sys.argv[1] if len(sys.argv)>1 else "dados/registrador"
Q=pd.read_csv("analise/padroes_quadros.csv"); Q["h"]=Q.hora.astype(int); R=[]
for f in glob.glob(f"{P}/Repetição */dl_rep*_camara*_*.csv"):
    rep,cam=f.split("dl_rep")[1].split("_camara"); x=pd.read_csv(f,sep=";",encoding="utf-8-sig"); x.columns=["t","T","UR"]; x["t"]=pd.to_datetime(x.t)
    x["data"]=x.t.dt.strftime("%Y-%m-%d"); x["h"]=x.t.dt.hour; x["box"]=int(cam[0]); R.append(x)
R=pd.concat(R).groupby(["data","box","h"])["T"].mean().reset_index(); out=[]
for nome,QQ in [("antes das 16 h",Q[Q.hora<16]),("todas as horas",Q)]:
    H=QQ.groupby(["idade","lote","data","box","trat","h"]).nnd_bl.median().reset_index().merge(R,on=["data","box","h"])
    H=H.groupby(["data","box"]).filter(lambda g: len(g)>=4)
    for (a,l,t),g in H.groupby(["idade","lote","trat"]): out.append(f"{nome} | {a} d {l} {t:12s} n={len(g)} T {g['T'].min():.1f}–{g['T'].max():.1f} rho={spearmanr(g['T'],g.nnd_bl)[0]:+.2f}")
open("analise/texto_s2.txt","w").write("\n".join(out)); print("\n".join(out))
