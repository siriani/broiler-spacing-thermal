"""Figura 3 do Artigo 2: (a–c) quadros reais típicos (20 d, L1, 10 h; quadro cuja % na zona do bebedouro é a mais próxima da média
do compartimento no dia), retificados ao plano da cama, com as zonas de 30 cm do bebedouro (azul) e do comedouro (laranja);
(d) % de aves na zona do bebedouro e do comedouro por tratamento: barras = média dos compartimento-datas, pontos = médias por lote,
tracejado = esperado ao acaso. Fontes: dados/quadros_figura3/selecao_zona.csv; analise/zonas_box_data.csv (figuras/fig_zonas.py)."""
import json,glob,numpy as np,pandas as pd,cv2,matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5})
src=open("analise/explora_espacamento.py").read(); ns={}; exec(src[:src.index("rows=[]")],ns); corretor=ns["corretor"]
CAL=json.load(open("dados/calibracao_v1.json")); K=json.load(open("dados/cantos_cam2_por_lote.json")); E=pd.read_csv("dados/equipamentos_por_data.csv"); S=3.0
T3=["frio","termoneutro","calor"]; LAB={"frio":"Cold","termoneutro":"Thermoneutral","calor":"Heat"}; CT={"frio":"#2b6cb0","termoneutro":"#2f855a","calor":"#c53030"}; MK={"rep1":"o","rep2":"s","rep3":"^"}
SEL=pd.read_csv("dados/quadros_figura3/selecao_zona.csv")
fig=plt.figure(figsize=(7.1,5.0)); gs=fig.add_gridspec(2,3,height_ratios=[1,1.05],hspace=.32,wspace=.06)
for j,t in enumerate(T3):
    r=SEL[SEL.trat==t].iloc[0]; cam=int(r.box); L,A=CAL[f"CAM{cam}"]["dim_cm"]; dst=np.float32([[0,0],[L,0],[L,A],[0,A]])
    H=cv2.getPerspectiveTransform(np.float32(K["rep1"]),dst).astype(float) if cam==2 else np.array(CAL[f"CAM{cam}"]["H_img_para_cm"],float)
    im=cv2.imread(f"dados/quadros_figura3/rep1_20_{t}_zona.png"); Wp=cv2.warpPerspective(im,np.diag([S,S,1])@H,(int(L*S),int(A*S)))
    ax=fig.add_subplot(gs[0,j]); ax.imshow(cv2.cvtColor(Wp,cv2.COLOR_BGR2RGB),extent=(0,L,A,0))
    d=pd.concat(pd.read_csv(x) for x in glob.glob(f"dados/medicao_v1/seg_todas_idades_v1_best/rep1_2026-02-24_CAM{cam}*.csv")); g=d[(d.video==r.video)&(d.quadro==r.quadro)]
    P0=g[["x_cm","y_cm"]].to_numpy(float); P=corretor(cam)(P0)
    e=E[(E.lote=="rep1")&(E.data=="2026-02-24")&(E.box==cam)]
    # zonas desenhadas na imagem (coordenadas sem correção de distorção, como a imagem retificada)
    for q in e.itertuples():
        ax.add_patch(Circle((q.x_cm,q.y_cm),30,fill=False,ec="#3182ce" if q.tipo=="bebedouro" else "#dd6b20",lw=1.6,ls="-" if q.tipo=="bebedouro" else "--"))
    Qb=e[e.tipo=="bebedouro"][["x_cm","y_cm"]].to_numpy(float); inz=np.linalg.norm(P[:,None]-Qb[None],axis=2).min(1)<=30
    ax.scatter(P0[:,0],P0[:,1],s=14,c=np.where(inz,"#3182ce","white"),edgecolors="#1a202c",linewidths=.6,zorder=3)
    ax.set_xlim(0,L); ax.set_ylim(A,0); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(CT[t]); sp.set_linewidth(2)
    ax.set_title(f"({'abc'[j]}) {dict(frio="Cold",termoneutro="TN",calor="Heat")[t]} · {inz.sum()} of 10 at the drinker",fontsize=7.2,color=CT[t],fontweight="bold",loc="left")
Z=pd.read_csv("analise/zonas_box_data.csv")
ax=fig.add_subplot(gs[1,:]); x=np.arange(3); w=0.36
for k,(col,off,lab,fc) in enumerate([("beb",-w/2,"Drinker zone","#4299e1"),("com",w/2,"Feeder zone (control)","#f6ad55")]):
    m=Z.groupby("trat")[col].mean().loc[T3]; ax.bar(x+off,m.values,w*0.92,color=fc,alpha=.85,label=lab,zorder=1)
    Lb=Z.groupby(["lote","trat"])[col].mean().unstack()[T3]
    for l,row in Lb.iterrows(): ax.scatter(x+off,row.values,marker=MK[l],s=24,c="#1a202c",zorder=3,edgecolors="white",linewidths=.4)
    ax.hlines(Z[col+"_esp"].mean(),x[0]+off-w/2,x[-1]+off+w/2,colors="0.35",linestyles=":",lw=1,zorder=2)
    for i,v in enumerate(m.values): ax.text(x[i]+off,1.5,f"{v:.0f}%",ha="center",fontsize=7,color="#1a202c",fontweight="bold")
ax.set_xticks(x,[LAB[t] for t in T3]); ax.set_ylabel("Birds within 30 cm (%)"); ax.set_ylim(0,58)
for l,mk in MK.items(): ax.scatter([],[],marker=mk,c="#1a202c",s=18,label=f"batch {l.replace('rep','L')}")
ax.plot([],[],":",c="0.35",label="random placement"); ax.legend(frameon=False,fontsize=6.5,ncol=3,loc="upper left")
ax.set_title("(d) Birds in the drinker and feeder zones (bars: mean of compartment-dates; symbols: batch means)",loc="left",fontsize=7.5,fontweight="bold")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
fig.savefig("figuras/fig_recursos.png",dpi=300,bbox_inches="tight"); fig.savefig("figuras/fig_recursos.pdf",bbox_inches="tight"); print("ok")
