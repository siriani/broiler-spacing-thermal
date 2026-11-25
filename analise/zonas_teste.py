"""Testes da % de aves na zona do bebedouro e do comedouro (<= 30 cm), mesmos 216 rearranjos por lote; saída analise/zonas_teste.txt"""
import itertools,numpy as np,pandas as pd
Z=pd.read_csv("analise/zonas_box_data.csv"); T3=["frio","termoneutro","calor"]
boxes=Z.groupby(["lote","box"]).trat.first(); lotes=sorted(Z.lote.unique()); allocs=[]
for perms in itertools.product(list(itertools.permutations(T3)),repeat=3):
    a={}
    for l,p in zip(lotes,perms):
        for b,t in zip(sorted(boxes[l].index),p): a[(l,b)]=t
    allocs.append(a)
def st(col,a):
    Y=Z.assign(t2=[a[(l,b)] for l,b in zip(Z.lote,Z.box)]); L=Y.groupby(["lote","t2"])[col].mean().unstack()
    V=Y.pivot_table(index="data",columns="t2",values=col); return (L.idxmax(axis=1)=="calor").sum(),(V["calor"]-V["termoneutro"]).mean()
out=[]
for col in ["beb","com"]:
    o=st(col,boxes.to_dict()); v=np.array([st(col,a) for a in allocs])
    out.append(f"{col}: médias {Z.groupby('trat')[col].mean().round(1).to_dict()}; esperado {Z[col+'_esp'].mean():.1f}; calor maior em {o[0]}/3 (p={(v[:,0]>=o[0]).mean():.3f}); calor−TN {o[1]:+.1f} pp (p unilateral={(v[:,1]>=o[1]).mean():.3f})")
    W=Z.pivot_table(index="data",columns="trat",values=col); out.append(f"   calor maior por data em {(W.idxmax(axis=1)=='calor').sum()}/7")
open("analise/zonas_teste.txt","w").write("\n".join(out)); print("\n".join(out))
