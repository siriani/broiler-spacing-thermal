"""Sensibilidades pedidas na revisão (rodada 1). Saída: analise/sensibilidade.txt e suplemento/Table_S2_retention.csv.
1) Retenção de quadros (exatamente 10, <10, >10 detecções) por compartimento-data. Quadros sem nenhuma detecção não estão
   nos arquivos de medição, por isso o denominador são os quadros com >= 1 detecção.
2) NND_BL com quadros de 9 a 11 detecções.  3) Janela comum aos três compartimentos em cada data; e só antes das 16 h.
4) NND dividida por um comprimento comum por data (mediana do eixo maior nos três compartimentos), em vez do eixo de cada ave.
5) Efeito do compartimento físico (descritivo).  6) Contraste de UR (Tabela 2).
Em todos: mediana por compartimento-data, média das datas por lote, ordem por lote e diferença média frio−TN e calor−TN."""
import glob,re,numpy as np,pandas as pd
from scipy.spatial import cKDTree
from datetime import datetime,timedelta
out=[]; P=lambda *a: out.append(" ".join(str(x) for x in a))
rows=[]; ret=[]
for f in sorted(glob.glob("dados/medicao_v1/seg_todas_idades_v1_best/rep*.csv")):
    d=pd.read_csv(f)
    if d.empty: continue
    v=d.video.iloc[0]; m=re.search(r"(\d{4})-?(\d{2})-?(\d{2})-?(\d{2})-?(\d{2})-?(\d{2})",v); t0=datetime(*map(int,m.groups()))
    for q,g in d.groupby("quadro"):
        n=len(g); t=t0+timedelta(seconds=float(g.t_s.iloc[0]))
        ret.append(dict(lote=g.lote.iloc[0],idade=int(g.idade_d.iloc[0]),data=g.data.iloc[0],box=int(g.box.iloc[0]),trat=g.tratamento.iloc[0],n=n))
        if not 9<=n<=11: continue
        P0=g[["x_cm","y_cm"]].to_numpy(float); ax=g.eixo_maior_cm.to_numpy(float); dd=cKDTree(P0).query(P0,k=2)[0][:,1]
        rows.append(dict(lote=g.lote.iloc[0],idade=int(g.idade_d.iloc[0]),data=g.data.iloc[0],box=int(g.box.iloc[0]),trat=g.tratamento.iloc[0],
                         n=n,hora=t.hour+t.minute/60,nbl=(dd/ax).mean(),ncm=dd.mean(),eixo_med=np.median(ax)))
R=pd.DataFrame(ret); F=pd.DataFrame(rows)
T=R.groupby(["lote","idade","data","box","trat"]).n.agg(total="size",exatos=lambda x:(x==10).mean(),menos=lambda x:(x<10).mean(),mais=lambda x:(x>10).mean()).reset_index()
T.round(3).to_csv("suplemento/Table_S2_retention.csv",index=False)
P("== 1. Retenção (proporção dos quadros com >=1 detecção)")
P(R.assign(ex=R.n==10,me=R.n<10,ma=R.n>10).groupby("trat")[["ex","me","ma"]].mean().round(3).to_string())
P(T.sort_values(["idade","lote","trat"]).round(2).to_string(index=False))
def resumo(F,col,nome):
    B=F.groupby(["lote","data","trat"])[col].median().reset_index()
    L=B.groupby(["lote","trat"])[col].mean().unstack()
    cold=(L.idxmin(axis=1)=="frio").sum(); heat=(L.idxmax(axis=1)=="calor").sum()
    W=B.pivot_table(index="data",columns="trat",values=col)
    P(f"-- {nome}: médias por lote (frio/TN/calor): "+"; ".join(f"{l.replace('rep','L')} {r.frio:.3f}/{r.termoneutro:.3f}/{r.calor:.3f}" for l,r in L.iterrows())
      +f" | frio menor em {cold}/3, calor maior em {heat}/3 | razão média por data frio/TN {(W.frio/W.termoneutro).mean():.3f}, calor/TN {(W.calor/W.termoneutro).mean():.3f}")
P("\n== 2–4. Sensibilidades do NND (mesma agregação do artigo)")
F10=F[F.n==10]; resumo(F10,"nbl","Primário (exatamente 10)")
resumo(F,"nbl","Quadros com 9–11 detecções")
jan=F10.groupby(["data","box"]).hora.agg(["min","max"]).reset_index().groupby("data").agg(ini=("min","max"),fim=("max","min"))
Fc=F10.merge(jan,on="data"); Fc=Fc[(Fc.hora>=Fc.ini)&(Fc.hora<=Fc.fim)]; resumo(Fc,"nbl","Janela comum aos três compartimentos")
P("   janela comum por data: "+"; ".join(f"{d} {int(r.ini)}:{int(r.ini%1*60):02d}–{int(r.fim)}:{int(r.fim%1*60):02d}" for d,r in jan.iterrows()))
resumo(F10[F10.hora<16],"nbl","Só antes das 16:00")
Lc=F10.groupby("data").eixo_med.median().rename("Lc"); F2=F10.merge(Lc,on="data"); F2["ncomum"]=F2.ncm/F2.Lc
resumo(F2,"ncomum","NND / comprimento comum da data")
P("   eixo maior mediano por tratamento relativo à data: "+", ".join(f"{t} {(F10[F10.trat==t].groupby('data').eixo_med.median()/Lc).mean():.3f}" for t in ["frio","termoneutro","calor"]))
P("\n== 5. Compartimento físico (descritivo; confundido com tratamento)")
B=F10.groupby(["data","box","trat"]).nbl.median().reset_index(); B["rel"]=B.nbl/B.groupby("data").nbl.transform("mean")
P(B.groupby("box").rel.mean().round(3).to_string())
P("\n== 6. UR (Tabela 2): diferenças em pontos percentuais")
U=pd.read_csv("analise/temperatura_janelas.csv").pivot_table(index=["idade","lote"],columns="trat",values="UR_janela")
P(pd.DataFrame({"frio-TN":U.frio-U.termoneutro,"calor-TN":U.calor-U.termoneutro}).round(0).to_string())
open("analise/sensibilidade.txt","w").write("\n".join(out)); print("\n".join(out))
