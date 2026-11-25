"""Posição de comedouro e bebedouros por data e câmera (Artigo 2).
Marcados manualmente (centro do prato, na cama) sobre a mediana de 15 quadros de cada gravação (dados/fundos_equip),
com grade de 50 px; precisão estimada ±10 px (~±3 cm). px -> cm pela mesma calibração das aves
(homografia da câmera; CAM2 com cantos por lote; correção de distorção em CAM1 e CAM3).
Saídas: dados/equipamentos_por_data.csv, video/confere_equipamentos.jpg, analise/dist_equip_box_data.csv."""
import json, glob, re, numpy as np, pandas as pd, cv2
src=open("analise/explora_espacamento.py").read(); ns={}; exec(src[:src.index("rows=[]")],ns); corretor=ns["corretor"]; CAL=ns["CAL"]
K=json.load(open("dados/cantos_cam2_por_lote.json"))
# (lote, data, cam): [(tipo, x_px, y_px)]  B = bebedouro, C = comedouro
PX={("rep1","2026-02-24",1):[("B",515,278),("B",533,555),("C",795,411)],
    ("rep1","2026-02-24",2):[("B",395,515),("B",420,795),("C",582,695)],
    ("rep1","2026-02-24",3):[("B",438,225),("B",431,510),("C",611,322)],
    ("rep1","2026-03-16",1):[("B",503,288),("B",518,573),("C",783,423)],
    ("rep1","2026-03-16",2):[("B",430,470),("B",440,760),("C",592,670)],
    ("rep1","2026-03-16",3):[("B",406,215),("B",443,510),("C",693,335)],
    ("rep2","2026-04-09",1):[("B",488,298),("B",515,568),("C",773,421)],
    ("rep2","2026-04-09",2):[("B",428,515),("B",618,700),("C",456,785)],
    ("rep2","2026-04-09",3):[("B",470,240),("B",470,510),("C",716,393)],
    ("rep2","2026-04-23",1):[("B",488,288),("B",511,568),("C",773,421)],
    ("rep2","2026-04-23",2):[("B",431,515),("B",623,705),("C",456,783)],
    ("rep2","2026-04-23",3):[("B",473,235),("B",463,520),("C",718,395)],
    ("rep2","2026-05-04",1):[("B",488,288),("B",513,568),("C",775,418)],
    ("rep2","2026-05-04",2):[("B",430,520),("B",628,710),("C",453,790)],
    ("rep2","2026-05-04",3):[("B",458,240),("B",463,525),("C",708,392)],
    ("rep3","2026-05-21",1):[("B",473,283),("C",483,548)],
    ("rep3","2026-05-21",2):[("B",385,400),("C",402,655)],
    ("rep3","2026-05-21",3):[("B",470,230),("C",472,495)],
    ("rep3","2026-06-18",1):[("B",461,253),("C",461,533)],
    ("rep3","2026-06-18",2):[("B",382,485),("C",412,795)],
    ("rep3","2026-06-18",3):[("B",448,230),("C",450,520)]}
def px2cm(lote,data,cam,P):
    C=CAL[f"CAM{cam}"]; L,A=C["dim_cm"]; dst=np.float32([[0,0],[L,0],[L,A],[0,A]])
    cp=np.float32(K.get(f"{lote}_{data}",K[lote])) if cam==2 else np.float32(C["cantos_px"])
    H=cv2.getPerspectiveTransform(cp,dst); q=cv2.perspectiveTransform(np.float32(P).reshape(-1,1,2),H).reshape(-1,2)
    return corretor(cam)(q.astype(float))
rows=[]; vis=[]
for (lote,data,cam),it in PX.items():
    P=[(x,y) for _,x,y in it]; Q=px2cm(lote,data,cam,P); L,A=CAL[f"CAM{cam}"]["dim_cm"]
    for (t,x,y),(xc,yc) in zip(it,Q): rows.append(dict(lote=lote,data=data,box=cam,tipo={"B":"bebedouro","C":"comedouro"}[t],x_px=x,y_px=y,x_cm=round(xc,1),y_cm=round(yc,1),xn=xc/L,yn=yc/A))
    im=cv2.imread(f"dados/fundos_equip/{lote}_{data}_CAM{cam}.png")
    for t,x,y in it: cv2.circle(im,(x,y),12,(255,128,0) if t=="B" else (0,0,255),3)
    cv2.putText(im,f"{lote} {data} CAM{cam}",(10,40),0,1.1,(0,255,255),2); vis.append(cv2.resize(im,(480,450)))
E=pd.DataFrame(rows); E.round(3).to_csv("dados/equipamentos_por_data.csv",index=False)
cv2.imwrite("video/confere_equipamentos.jpg",np.vstack([np.hstack(vis[i:i+3]) for i in range(0,21,3)]),[cv2.IMWRITE_JPEG_QUALITY,80])
# distância de cada ave ao bebedouro e ao comedouro mais próximos (quadros com 10 aves), média por box-data
out=[]
for f in sorted(glob.glob("dados/medicao_v1/seg_todas_idades_v1_best/rep*.csv")):
    d=pd.read_csv(f)
    if d.empty: continue
    d=d[d.n_aves_quadro==10]; cam=int(d.box.iloc[0]); lote=d.lote.iloc[0]; data=d.data.iloc[0]
    e=E[(E.lote==lote)&(E.data==data)&(E.box==cam)]; P=corretor(cam)(d[["x_cm","y_cm"]].to_numpy(float))
    for tipo in ["bebedouro","comedouro"]:
        Q=e[e.tipo==tipo][["x_cm","y_cm"]].to_numpy(float); dist=np.linalg.norm(P[:,None]-Q[None],axis=2).min(1)
        d[f"d_{tipo}"]=dist; d[f"perto_{tipo}"]=dist<25
    out.append(d.assign(n=1)[["lote","idade_d","data","box","tratamento","d_bebedouro","d_comedouro","perto_bebedouro","perto_comedouro"]])
D=pd.concat(out); B=D.groupby(["lote","idade_d","data","box","tratamento"]).mean(numeric_only=True).reset_index()
B.round(3).to_csv("analise/dist_equip_box_data.csv",index=False)
print(E.round(1).to_string(index=False)); print()
print(B.groupby("tratamento")[["d_bebedouro","d_comedouro","perto_bebedouro","perto_comedouro"]].mean().round(3))
print(B.groupby(["lote","tratamento"])[["d_bebedouro","d_comedouro"]].mean().unstack().round(1))
