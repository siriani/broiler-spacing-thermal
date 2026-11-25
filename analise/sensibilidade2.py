"""Rodada 2 da revisão: leave-one-date-out, ajuste por compartimento, razão por data × contraste térmico,
intervalo por inversão do teste de aleatorização, estabilidade (pares × ímpares), faixas de temperatura e retenção <= 8.
Saída: analise/sensibilidade2.txt."""
import itertools,numpy as np,pandas as pd,glob
out=[]; P=lambda *a: out.append(" ".join(str(x) for x in a))
B=pd.read_csv("analise/box_data.csv"); T3=["frio","termoneutro","calor"]
W=B.pivot_table(index=["lote","data","idade"],columns="trat",values="nnd_bl").reset_index()
def order_meandiff(W):
    L=W.groupby("lote")[T3].mean(); cold=(L.idxmin(axis=1)=="frio").sum(); heat=(L.idxmax(axis=1)=="calor").sum()
    return cold,heat,(W.frio-W.termoneutro).mean(),(W.frio/W.termoneutro).mean(),(W.calor/W.termoneutro).mean()
P("== Leave-one-date-out (ordem por lote; diferença média frio−TN; razões frio/TN e calor/TN)")
for i,r in W.iterrows():
    c,h,d,rc,rh=order_meandiff(W.drop(i)); P(f"sem {r.idade} d {r.lote}: frio menor {c}/3, calor maior {h}/3, frio−TN {d:+.3f}, frio/TN {rc:.3f}, calor/TN {rh:.3f}")
# compartimento
import statsmodels.formula.api as smf
m=smf.ols("nnd_bl ~ C(data) + C(box) + C(trat, Treatment('termoneutro'))",data=B).fit()
P("\n== OLS descritivo nnd_bl ~ data + compartimento + tratamento (21 pontos)")
for k,v in m.params.items():
    if "box" in k or "trat" in k: P(f"{k}: {v:+.4f}")
rel=B.assign(rel=B.nnd_bl-B.groupby("data").nnd_bl.transform("mean")).groupby("box").rel.mean(); P("compartimento − média da data (BL):",rel.round(3).to_dict())
# razão por data x contraste
Tj=pd.read_csv("analise/temperatura_janelas.csv").pivot_table(index=["idade","lote"],columns="trat",values="T_janela")
Wd=W.set_index(["idade","lote"])
P("\n== Razão por data × contraste térmico")
for (a,l) in Wd.index:
    P(f"{a} d {l}: frio/TN {Wd.loc[(a,l),'frio']/Wd.loc[(a,l),'termoneutro']:.3f} (ΔT {Tj.loc[(a,l),'frio']-Tj.loc[(a,l),'termoneutro']:+.1f}); calor/TN {Wd.loc[(a,l),'calor']/Wd.loc[(a,l),'termoneutro']:.3f} (ΔT {Tj.loc[(a,l),'calor']-Tj.loc[(a,l),'termoneutro']:+.1f})")
rc=[Wd.loc[k,'frio']/Wd.loc[k,'termoneutro'] for k in Wd.index]; dc=[Tj.loc[k,'frio']-Tj.loc[k,'termoneutro'] for k in Wd.index]
rh=[Wd.loc[k,'calor']/Wd.loc[k,'termoneutro'] for k in Wd.index]; dh=[Tj.loc[k,'calor']-Tj.loc[k,'termoneutro'] for k in Wd.index]
P(f"Spearman razão × ΔT: frio {pd.Series(rc).corr(pd.Series(dc),method='spearman'):+.2f}; calor {pd.Series(rh).corr(pd.Series(dh),method='spearman'):+.2f} (7 pontos)")
# estabilidade par/ímpar
P("\n== Estabilidade da mediana por compartimento-data: quadros pares × ímpares")
F=pd.read_csv("analise/quadros_10aves.csv")
F["k"]=F.groupby(["data","box"]).cumcount()%2
H=F.groupby(["data","box","k"]).nnd_bl.median().unstack()
P(f"diferença absoluta par−ímpar: mediana {(H[0]-H[1]).abs().median():.4f}, máxima {(H[0]-H[1]).abs().max():.4f} BL; correlação {H[0].corr(H[1]):.3f}")
# faixas de temperatura
P("\n== Faixa de temperatura nos quadros analisados (mín–máx, °C)")
J=pd.read_csv("analise/temperatura_janelas.csv")
for (a,l),g in J.groupby(["idade","lote"]):
    P(f"{a} d {l}: "+"; ".join(f"{r.trat} {r.T_min:.1f}–{r.T_max:.1f}" for r in g.itertuples()))
# retenção <=8
S=pd.read_csv("suplemento/Table_S2_retention.csv")
R=[]
for f in glob.glob("dados/medicao_v1/seg_todas_idades_v1_best/rep*.csv"):
    d=pd.read_csv(f)
    if d.empty: continue
    n=d.groupby("quadro").size(); R.append(dict(data=d.data.iloc[0],box=int(d.box.iloc[0]),ate8=(n<=8).mean()))
S=S.merge(pd.DataFrame(R).groupby(["data","box"]).ate8.mean().reset_index(),on=["data","box"]); S.round(3).to_csv("suplemento/Table_S2_retention.csv",index=False)
P("\n== Quadros com <= 8 detecções (fração):"); P(S[["idade","lote","trat","exatos","ate8"]].sort_values(["idade","lote"]).round(3).to_string(index=False))
open("analise/sensibilidade2.txt","w").write("\n".join(out)); print("\n".join(out))
