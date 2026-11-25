"""Figura 3 (proposta): % de aves na zona do bebedouro e na zona do comedouro (centro a <= 30 cm), por tratamento.
Pontos = compartimento-datas (formas = lotes); traço = média; linha tracejada = % esperada ao acaso (área da zona / área da cama).
Posições corrigidas de distorção (CAM1/CAM3), quadros com 10 aves. Saída também: analise/zonas_box_data.csv"""
import glob,numpy as np,pandas as pd,matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5})
src=open("analise/explora_espacamento.py").read(); ns={}; exec(src[:src.index("rows=[]")],ns); corretor=ns["corretor"]; CAL=ns["CAL"]
E=pd.read_csv("dados/equipamentos_por_data.csv"); R0=30.0; rng=np.random.default_rng(1)
T3=["frio","termoneutro","calor"]; LAB={"frio":"Cold","termoneutro":"Thermo-\nneutral","calor":"Heat"}; CT={"frio":"#2b6cb0","termoneutro":"#2f855a","calor":"#c53030"}; MK={"rep1":"o","rep2":"s","rep3":"^"}
rows=[]
for f in sorted(glob.glob("dados/medicao_v1/seg_todas_idades_v1_best/rep*.csv")):
    d=pd.read_csv(f)
    if d.empty: continue
    d=d[d.n_aves_quadro==10]; cam=int(d.box.iloc[0]); L,A=CAL[f"CAM{cam}"]["dim_cm"]; lote,data=d.lote.iloc[0],d.data.iloc[0]
    P=corretor(cam)(d[["x_cm","y_cm"]].to_numpy(float)); U=np.c_[rng.uniform(0,L,100000),rng.uniform(0,A,100000)]; r=dict(lote=lote,data=data,box=cam,trat=d.tratamento.iloc[0])
    for tipo,k in [("bebedouro","beb"),("comedouro","com")]:
        Q=E[(E.lote==lote)&(E.data==data)&(E.box==cam)&(E.tipo==tipo)][["x_cm","y_cm"]].to_numpy(float)
        r[k]=100*(np.linalg.norm(P[:,None]-Q[None],axis=2).min(1)<=R0).sum(); r["n"]=len(P); r[k+"_esp"]=100*(np.linalg.norm(U[:,None]-Q[None],axis=2).min(1)<=R0).mean()
    rows.append(r)
Z=pd.DataFrame(rows); Z=Z.groupby(["lote","data","box","trat"]).agg(beb=("beb","sum"),com=("com","sum"),n=("n","sum"),beb_esp=("beb_esp","mean"),com_esp=("com_esp","mean")).reset_index(); Z["beb"]/=Z.n; Z["com"]/=Z.n; Z.round(2).to_csv("analise/zonas_box_data.csv",index=False)
fig,ax=plt.subplots(1,2,figsize=(6.2,2.9),sharey=True)
for a,(k,tt) in zip(ax,[("beb","(a) Drinker zone"),("com","(b) Feeder zone (control)")]):
    for i,t in enumerate(T3):
        g=Z[Z.trat==t]; xs=i+np.linspace(-.18,.18,len(g))
        for x,(_,rr) in zip(xs,g.iterrows()): a.scatter(x,rr[k],marker=MK[rr.lote],s=22,c=CT[t],alpha=.85,edgecolors="white",linewidths=.4,zorder=3)
        a.plot([i-.3,i+.3],[g[k].mean()]*2,c=CT[t],lw=2.2,zorder=4)
    a.axhline(Z[k+"_esp"].mean(),c="0.45",ls="--",lw=1); a.text(2.45,Z[k+"_esp"].mean(),"random",fontsize=6.5,color="0.4",va="bottom",ha="right")
    a.set_xticks(range(3),[LAB[t] for t in T3]); a.set_title(tt,loc="left",fontweight="bold"); a.set_xlim(-.5,2.5)
    for sp in ["top","right"]: a.spines[sp].set_visible(False)
ax[0].set_ylabel("Birds within 30 cm (%)")
for l,m in MK.items(): ax[1].scatter([],[],marker=m,c="0.3",s=18,label=l.replace("rep","L"))
ax[1].legend(frameon=False,fontsize=6.5,loc="upper right",title="batch",title_fontsize=6.5)
plt.tight_layout(); plt.savefig("figuras/fig_zonas.png",dpi=300); plt.savefig("figuras/fig_zonas.pdf")
print(Z.groupby("trat")[["beb","beb_esp","com","com_esp"]].mean().round(1)); print(Z.groupby(["lote","trat"]).beb.mean().unstack().round(1))
