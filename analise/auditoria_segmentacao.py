"""Auditoria da segmentação v1 por tratamento nos 48 quadros de teste do Artigo 1 (máscaras corrigidas à mão).
Por quadro: contagem, recall/precisão (IoU>=0,5), IoU, e NND_BL calculada das máscaras manuais × do modelo (pixels; CAM2 com x×2)."""
import json,glob,os,sys,numpy as np,cv2,pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from ultralytics import YOLO
TR={(1,1):"TN",(1,2):"Heat",(1,3):"Cold",(2,1):"TN",(2,2):"Cold",(2,3):"Heat",(3,1):"Heat",(3,2):"TN",(3,3):"Cold"}
d=json.load(open("cvat_corrigido_t12.json")); names=sorted(os.path.basename(p) for p in glob.glob("cvat/*.jpg"))
G={}
for s in d["shapes"]: G.setdefault(s["frame"],[]).append(np.array(s["points"]).reshape(-1,2))
m=YOLO(sys.argv[1]); rows=[]
def geom(masks,sx):
    c=[];ax=[]
    for mk in masks:
        cs,_=cv2.findContours(mk.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not cs: continue
        k=max(cs,key=cv2.contourArea).reshape(-1,2).astype(np.float32); k[:,0]*=sx
        if len(k)<5: continue
        (x,y),(a,b),_=cv2.fitEllipse(k); c.append((x,y)); ax.append(max(a,b))
    c=np.array(c);ax=np.array(ax)
    if len(c)<2: return np.nan
    dd,_=cKDTree(c).query(c,k=2); return (dd[:,1]/ax).mean()
for f,n in enumerate(names):
    img=cv2.imread("cvat/"+n); h,w=img.shape[:2]
    g=[]
    for p in G.get(f,[]):
        mk=np.zeros((h,w),np.uint8); cv2.fillPoly(mk,[p.astype(np.int32)],1); g.append(mk>0)
    r=m.predict(img,imgsz=1024,conf=0.25,verbose=False,retina_masks=True)[0]
    P=[] if r.masks is None else [(x>0.5) for x in r.masks.data.cpu().numpy()]
    P=[x if x.shape==(h,w) else cv2.resize(x.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0 for x in P]
    I=np.array([[(a&b).sum()/max((a|b).sum(),1) for b in g] for a in P]) if P and g else np.zeros((0,0))
    tp=0;ious=[]
    if I.size:
        ri,ci=linear_sum_assignment(-I); pr=[(i,j) for i,j in zip(ri,ci) if I[i,j]>=0.5]; tp=len(pr); ious=[I[i,j] for i,j in pr]
    age=int(n.split("_d")[1][:2]); L=int(n.split("_L")[1][0]); C=int(n.split("_C")[1][0]); sx=2.0 if C==2 else 1.0
    rows.append(dict(img=n,idade=age,lote=L,box=C,trat=TR[(L,C)],gt=len(g),pred=len(P),tp=tp,iou=np.mean(ious) if ious else np.nan,
                     nnd_gt=geom(g,sx),nnd_pred=geom(P,sx)))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv("auditoria_v1.csv",index=False)
