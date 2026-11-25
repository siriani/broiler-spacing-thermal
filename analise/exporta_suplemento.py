"""Writes the supplementary tables in English (as published): Table S2 (frame retention), Table S3 (recording windows,
temperature ranges and specific enthalpy of the air) and Table S6 (drinker and feeder zones). Run after the other scripts."""
import numpy as np,pandas as pd
mt={"frio":"Cold","termoneutro":"TN","calor":"Heat"}; ml=lambda x:x.replace("rep","L")
S2=pd.read_csv("suplemento/Table_S2_retention.csv")
if "lote" in S2.columns:
    S2=S2.assign(lote=S2.lote.map(ml),trat=S2.trat.map(mt),box=S2.box.map(lambda b:f"C{b}")).rename(columns={"lote":"Batch","idade":"Age (d)","data":"Date","box":"Compartment","trat":"Treatment","total":"Frames with ≥1 detection","exatos":"Fraction with exactly 10","menos":"Fraction <10","mais":"Fraction >10","ate8":"Fraction ≤8"})
    S2.to_csv("suplemento/Table_S2_retention.csv",index=False)
Z=pd.read_csv("analise/zonas_box_data.csv")
Z=Z.assign(lote=Z.lote.map(ml),trat=Z.trat.map(mt),box=Z.box.map(lambda b:f"C{b}")).rename(columns={"lote":"Batch","data":"Date","box":"Compartment","trat":"Treatment","beb":"Drinker zone (%)","com":"Feeder zone (%)","n":"Bird positions","beb_esp":"Drinker zone expected (%)","com_esp":"Feeder zone expected (%)"})
Z.round(2).to_csv("suplemento/Table_S6_resources.csv",index=False)
J=pd.read_csv("analise/temperatura_janelas.csv")
def h(T,UR,P=93.9):   # specific enthalpy (kJ/kg dry air); Campinas ~680 m a.s.l.
    es=0.61078*np.exp(17.27*T/(T+237.3)); e=UR/100*es; w=0.622*e/(P-e); return 1.006*T+w*(2501+1.86*T)
J["h"]=h(J.T_janela,J.UR_janela); rows=[]
for (a,l),g in J.groupby(["idade","lote"]):
    g=g.set_index("trat"); r={"Age (d)":a,"Batch":ml(l)}
    for k,v in mt.items(): r[f"{v} window"]=f"{g.loc[k,'ini']}–{g.loc[k,'fim']}"
    for k,v in mt.items(): r[f"{v} min–max (°C)"]=f"{g.loc[k,'T_min']:.1f}–{g.loc[k,'T_max']:.1f}"
    for k,v in mt.items(): r[f"{v} enthalpy (kJ/kg dry air)"]=round(g.loc[k,"h"],1)
    rows.append(r)
pd.DataFrame(rows).to_csv("suplemento/Table_S3_temperature_ranges.csv",index=False); print("supplementary tables written")
