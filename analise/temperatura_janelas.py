"""Temperatura e UR do registrador (Elitech, 0,20 m, 5 min; CSV do pacote do Igor de 24/11/2026) nos quadros analisados.
Cada quadro (padroes_quadros.csv: data, box, hora) recebe a leitura mais próxima do registrador do seu compartimento;
média por box-data (ponderada pelos quadros) e, para comparação, a média da sessão (8 h–16 h).
Saídas: analise/temperatura_janelas.csv."""
import glob,pandas as pd,numpy as np
P="dados/registrador"
L=[]
for f in glob.glob(f"{P}/Repetição */dl_rep*_camara*_*.csv"):
    rep,cam=f.split("dl_rep")[1].split("_camara"); cam=int(cam[0]); d=pd.read_csv(f,sep=";",encoding="utf-8-sig")
    d.columns=["t","T","UR"]; d["t"]=pd.to_datetime(d.t); d["lote"]=f"rep{rep}"; d["box"]=cam; L.append(d)
R=pd.concat(L).sort_values("t")
Q=pd.read_csv("analise/padroes_quadros.csv")
Q["t"]=pd.to_datetime(Q.data)+pd.to_timedelta(Q.hora,unit="h")
out=[]
for (data,box),g in Q.groupby(["data","box"]):
    r=R[(R.box==box)&(R.t.dt.strftime("%Y-%m-%d")==data)].set_index("t")
    if r.empty: print("sem registro",data,box); continue
    idx=r.index.get_indexer(g.t.sort_values(),method="nearest"); rr=r.iloc[idx]
    s=r.between_time("08:00","16:00")
    out.append(dict(data=data,lote=g.lote.iloc[0],idade=g.idade.iloc[0],box=box,trat=g.trat.iloc[0],ini=g.t.min().strftime("%H:%M"),fim=g.t.max().strftime("%H:%M"),
        T_janela=rr["T"].mean(),UR_janela=rr.UR.mean(),T_min=rr["T"].min(),T_max=rr["T"].max(),T_sessao=s["T"].mean(),UR_sessao=s.UR.mean()))
O=pd.DataFrame(out).sort_values(["idade","lote","trat"]); O.round(2).to_csv("analise/temperatura_janelas.csv",index=False)
print(O.round(1).to_string(index=False))
W=O.pivot_table(index=["idade","lote"],columns="trat",values="T_janela")
W["frio-TN"]=W.frio-W.termoneutro; W["calor-TN"]=W.calor-W.termoneutro; print(W.round(1)); print(W[["frio-TN","calor-TN"]].mean().round(2))
