"""Figura do delineamento (Artigo 2): (a) croqui isométrico de um compartimento; (b) planta dos 9 compartimentos com
comedouro e bebedouros medidos no vídeo; (c) calendário das sessões de 8 h e das datas filmadas.
Fontes: dados/equipamentos_por_data.csv (analise/equipamentos.py); sessões: relatório do Igor (Tabela 11), dados/igor/EXTRATO.md.
Itens sem posição registrada (desumidificadores, registrador; aquecedor pela foto do Igor; split na parede de azulejo, por decisão do Allan) aparecem tracejados, em posição ASSUMIDA."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from scipy.spatial import ConvexHull
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7})
E=pd.read_csv("dados/equipamentos_por_data.csv")
ULT={"rep1":"2026-03-16","rep2":"2026-05-04","rep3":"2026-06-18"}
TR={("rep1",1):"TN",("rep1",2):"Heat",("rep1",3):"Cold",("rep2",1):"TN",("rep2",2):"Cold",("rep2",3):"Heat",("rep3",1):"Heat",("rep3",2):"TN",("rep3",3):"Cold"}
CT={"Cold":"#2b6cb0","TN":"#2f855a","Heat":"#c53030"}; AS="#c05621"
SES={"L1":[20,23,26,29,33,36,40],"L2":[15,19,22,26,29,33,36,40],"L3":[8,12,15,19,22,26,29,33,36,40]}
REC={"L1":[20,40],"L2":[15,29,40],"L3":[8,36]}
# ---------- (a) isométrico
RA,RB,RZ=160,140,230; OA,OB=2,5; C30,S30=np.cos(np.pi/6),np.sin(np.pi/6)
def P(a,b,z): return ((b-a)*C30, z-(a+b)*S30)
def poly(ax,pts,fc,ec="#333",ls="-",lw=.5,z=1): ax.add_patch(Polygon([P(*p) for p in pts],closed=True,fc=fc,ec=ec,ls=ls,lw=lw,zorder=z))
def cyl(ax,a,b,r,z0,z1,fc,z):
    t=np.linspace(0,2*np.pi,48); bot=[P(a+r*np.cos(s),b+r*np.sin(s),z0) for s in t]; top=[P(a+r*np.cos(s),b+r*np.sin(s),z1) for s in t]
    pts=np.array(bot+top); h=ConvexHull(pts); ax.add_patch(Polygon(pts[h.vertices],fc=fc,ec="#333",lw=.5,zorder=z)); ax.add_patch(Polygon(top,fc=fc,ec="#333",lw=.5,zorder=z+.01))
def boxw(ax,a0,a1,b0,b1,z0,z1,fc,z,ls="-",ec="#333"):
    for f in [[(a0,b0,z1),(a1,b0,z1),(a1,b1,z1),(a0,b1,z1)],[(a1,b0,z0),(a1,b1,z0),(a1,b1,z1),(a1,b0,z1)],[(a0,b1,z0),(a1,b1,z0),(a1,b1,z1),(a0,b1,z1)]]: poly(ax,f,fc,ec=ec,ls=ls,z=z)
def lab(ax,a,b,z,s,dx,dy,c="#222"):
    x,y=P(a,b,z); ax.annotate(s,(x,y),(x+dx,y+dy),fontsize=6.5,color=c,ha="center",va="center",zorder=60,arrowprops=dict(arrowstyle="-",lw=.5,color=c))
fig=plt.figure(figsize=(7.1,8.2)); gs=fig.add_gridspec(2,2,height_ratios=[1.05,1],width_ratios=[1.15,1],hspace=.28,wspace=.08)
ax=fig.add_subplot(gs[0,0]); L,A=156,130
poly(ax,[(0,0,0),(RA,0,0),(RA,RB,0),(0,RB,0)],"#d5d5d5",z=1)
poly(ax,[(0,0,0),(0,RB,0),(0,RB,RZ),(0,0,RZ)],"#eef3f8",z=2)
for zz in range(20,RZ,20): ax.plot(*zip(P(0,0,zz),P(0,RB,zz)),c="#d5e0ea",lw=.35,zorder=2.1)
for bb in range(20,RB,20): ax.plot(*zip(P(0,bb,0),P(0,bb,RZ)),c="#d5e0ea",lw=.35,zorder=2.1)
poly(ax,[(0,0,0),(RA,0,0),(RA,0,RZ),(0,0,RZ)],"#c9cfd6",z=2)
bc=OB+65; t=np.linspace(0,2*np.pi,60)
ax.add_patch(Polygon([P(0,bc+20*np.cos(s),80+20*np.sin(s)) for s in t],fc="#5a7bd5",ec="#1f3a93",lw=.8,zorder=3))
poly(ax,[(0,bc-38,168),(0,bc+38,168),(0,bc+38,196),(0,bc-38,196)],"#ffffff",ec="#333",lw=.8,z=3)
boxw(ax,0,22,OB+108,OB+132,118,121,"#c8a165",4); boxw(ax,0,14,OB+110,OB+130,121,150,"#f2f2f2",4.1)
boxw(ax,0,10,8,20,40,60,"#d6ecfa",4,ls="--",ec=AS); boxw(ax,0,10,24,36,40,60,"#d6ecfa",4,ls="--",ec=AS)
boxw(ax,0,3,OB+118,OB+126,18,24,"#4a5568",4,ls="--",ec=AS)
poly(ax,[(OA,OB,.5),(OA+L,OB,.5),(OA+L,OB+A,.5),(OA,OB+A,.5)],"#e8cf8f",ec="#a0824a",z=5)
e=E[(E.lote=="rep1")&(E.data==ULT["rep1"])&(E.box==1)].copy(); e["a"]=OA+(L-e.x_cm); e["b"]=OB+e.y_cm; e=e.assign(d=e.a+e.b).sort_values("d")
for k,r in enumerate(e.itertuples()):
    zz=10+k; ax.plot(*zip(P(r.a,r.b,40 if r.tipo=="comedouro" else 30),P(r.a,r.b,RZ)),c="#8a94a3",lw=.5,zorder=zz-.5)
    if r.tipo=="comedouro": cyl(ax,r.a,r.b,13.5,1,7,"#dd6b20",zz); cyl(ax,r.a,r.b,6,7,40,"#f6ad55",zz+.1); fe=(r.a,r.b)
    else: cyl(ax,r.a,r.b,11,4,8,"#e53e3e",zz); cyl(ax,r.a,r.b,8,8,30,"#f7fafc",zz+.1); dr=(r.a,r.b)
for q in [[(RA,0,0),(RA,0,RZ),(RA,RB,RZ),(RA,RB,0)],[(0,RB,0),(0,RB,RZ),(RA,RB,RZ),(RA,RB,0)]]: ax.plot(*zip(*[P(*p) for p in q]),c="#888",lw=.6,ls=":",zorder=30)
ax.plot(*zip(*[P(RA,OB+35,0),P(RA,OB+35,195),P(RA,OB+105,195),P(RA,OB+105,0)]),c="#444",lw=.9,zorder=30)
ax.add_patch(Polygon([P(RA/2-5,RB/2-5,214),P(RA/2+5,RB/2-5,214),P(RA/2+5,RB/2+5,214),P(RA/2-5,RB/2+5,214)],fc="#2d3748",zorder=31))
ax.plot(*zip(P(RA/2,RB/2,212),P(RA/2,RB/2,0)),c="#2d3748",lw=.5,ls=(0,(1,3)),zorder=29)
lab(ax,RA/2,RB/2,214,"camera, 2.12 m",-45,22)
lab(ax,0,bc,182,"split A/C",40,22); lab(ax,7,OB+120,150,"fan heater on shelf",50,14); lab(ax,5,22,60,"dehumidifiers*",40,40,AS); lab(ax,2,OB+122,21,"logger, 0.20 m*",48,-12,AS)
lab(ax,0,bc,80,"exhaust fan, 0.80 m",52,-22); lab(ax,RA,OB+70,120,"door (front wall)",-38,-6)
lab(ax,fe[0],fe[1],40,"tube feeder",-10,-48); lab(ax,dr[0],dr[1],20,"bell drinker",-40,-26)
ax.set_aspect("equal"); ax.autoscale_view(); ax.axis("off"); ax.set_title("(a) One compartment (L1, C1)",loc="left",fontsize=8,fontweight="bold")
ax.text(0.0,-0.02,"* position not recorded, assumed",transform=ax.transAxes,fontsize=6,color=AS)
# ---------- (c) calendário
ax=fig.add_subplot(gs[0,1])
for i,l in enumerate(["L1","L2","L3"]):
    y=2-i; ax.plot([1,42],[y,y],c="#bbb",lw=1,zorder=1)
    for s in SES[l]: ax.add_patch(Rectangle((s-.45,y-.18),.9,.36,fc="#f6ad55",ec="none",zorder=2))
    for s in REC[l]: ax.plot(s,y+.36,marker="v",ms=5,c="#222",zorder=3)
    ax.text(-.5,y,l,ha="right",va="center",fontsize=7.5,fontweight="bold")
ax.set_xlim(-3,43); ax.set_ylim(-1.2,3.1); ax.set_yticks([]); ax.set_xticks([1,8,15,22,29,36,42]); ax.set_xlabel("Age (d)")
for sp in ["left","right","top"]: ax.spines[sp].set_visible(False)
ax.add_patch(Rectangle((1,-.95),1.2,.3,fc="#f6ad55",ec="none")); ax.text(2.8,-.8,"8-h treatment session (08:00–16:00)",va="center",fontsize=6.5)
ax.plot(1.6,-.35,marker="v",ms=5,c="#222"); ax.text(2.8,-.35,"recording analysed",va="center",fontsize=6.5)
ax.text(1,2.75,"Between sessions all compartments were thermoneutral",fontsize=6.5,color="#555")
ax.set_title("(c) Treatment sessions and recordings",loc="left",fontsize=8,fontweight="bold")
# ---------- (b) plantas
sub=gs[1,:].subgridspec(3,3,hspace=.35,wspace=.05)
for i,lote in enumerate(["rep1","rep2","rep3"]):
    for j,cam in enumerate([1,2,3]):
        ax=fig.add_subplot(sub[i,j]); A=134 if cam==3 else 130; tr=TR[(lote,cam)]
        ax.add_patch(Rectangle((0,0),156,A,fc="#f3e3b5",ec=CT[tr],lw=2))
        ax.plot([-4,-4],[0,A],c="#444",lw=2); ax.plot([-4,-4],[30,100],c="white",lw=2.2); ax.text(-9,A/2,"door",rotation=90,va="center",ha="center",fontsize=5.5,color="#444")
        ax.plot([160,160],[0,A],c="#7f9cc2",lw=2); ax.add_patch(Circle((163,65),7,fc="#5a7bd5",ec="none")); ax.text(173,A/2,"fan",rotation=90,va="center",ha="center",fontsize=5.5,color="#1f3a93")
        e=E[(E.lote==lote)&(E.data==ULT[lote])&(E.box==cam)]
        for r in e.itertuples():
            if r.tipo=="comedouro": ax.add_patch(Circle((r.x_cm,r.y_cm),13.5,fc="#dd6b20",ec="#7b341e",lw=.6))
            else: ax.add_patch(Circle((r.x_cm,r.y_cm),11,fc="#f7fafc",ec="#e53e3e",lw=1.6))
        if i==2 and j==0: ax.plot([0,50],[A+10,A+10],c="k",lw=1.2); ax.text(25,A+20,"50 cm",ha="center",va="center",fontsize=5.5)
        ax.set_xlim(-14,178); ax.set_ylim(A+26,-4); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{lote.replace('rep','L')} · C{cam} · {tr}",fontsize=7,color=CT[tr],fontweight="bold",pad=2)
fig.canvas.draw(); bb=fig.axes[2].get_position(); fig.text(bb.x0,bb.y1+0.035,"(b) Plan of the nine compartments (camera view; litter 156 × 130–134 cm; feeder and drinkers at the last recording)",fontsize=8,fontweight="bold")
fig.savefig("figuras/fig_setup.png",dpi=300,bbox_inches="tight"); fig.savefig("figuras/fig_setup.pdf",bbox_inches="tight"); print("ok")
