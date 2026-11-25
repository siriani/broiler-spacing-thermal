"""Análise confirmatória do Artigo 2 (H1: frio < TN; H2: calor > TN) em NND por comprimento de corpo.

Entradas: padroes_box_data.csv (mediana por box-data, NND em comprimentos de corpo),
          box_data.csv (NND corrigida em cm, distância à parede etc.),
          padroes_quadros.csv (por quadro, com hora).
Saídas:   stats_confirmatoria.txt e stats_por_data.csv.

Unidade experimental: box dentro do lote. Aleatorização exata respeitando o sorteio real (tratamentos
permutados entre os 3 boxes de cada lote, mesma permutação para todas as datas do lote; 6^3 = 216 alocações).
Duas estatísticas: (1) ordem por lote (nº de lotes com frio menor / calor maior; prevista no PLANO_ARTIGO2.md)
e (2) diferença média nas datas (frio − TN; calor − TN; tamanho do efeito). A análise NÃO foi pré-registrada:
uma exploração dos mesmos dados veio antes (ver fatos_resultados.md). Complemento descritivo: modelo misto
resposta ~ trat + data + (1 | box no lote). Dados: dados/medicao_v1 (CAM2 recalibrada em L1/L2).
"""
import itertools, os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

H = os.path.dirname(os.path.abspath(__file__))
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

pb = pd.read_csv(f"{H}/padroes_box_data.csv")
bd = pd.read_csv(f"{H}/box_data.csv")
df = pb.merge(bd[["lote", "data", "box", "nnd", "nnd_sem", "parede", "disp", "ocup"]], on=["lote", "data", "box"])
TR = ["frio", "termoneutro", "calor"]

def contrastes(d, col, alloc=None):
    """alloc: dict lote -> dict box -> trat (permutação). Retorna (frio−TN, calor−TN) médios nas datas."""
    d = d.copy()
    if alloc is not None:
        d["trat"] = [alloc[l][b] for l, b in zip(d.lote, d.box)]
    w = d.pivot_table(index="data", columns="trat", values=col)
    return (w.frio - w.termoneutro).mean(), (w.calor - w.termoneutro).mean()

def ordem_lote(d, alloc=None):
    """Nº de lotes em que o frio tem a menor média (nas datas do lote) e o calor a maior."""
    d = d.copy()
    if alloc is not None:
        d["trat"] = [alloc[l][b] for l, b in zip(d.lote, d.box)]
    m = d.groupby(["lote", "trat"]).nnd_bl.mean().unstack()
    return int((m.idxmin(axis=1) == "frio").sum()), int((m.idxmax(axis=1) == "calor").sum())

def aleatorizacao(col, rotulo, bilateral=False):
    obs = contrastes(df, col)
    lotes = sorted(df.lote.unique())
    reais = {l: dict(df[df.lote == l].groupby("box").trat.first()) for l in lotes}
    perms_l = {}
    for l in lotes:
        boxes = sorted(reais[l]); trats = [reais[l][b] for b in boxes]
        perms_l[l] = [dict(zip(boxes, pr)) for pr in itertools.permutations(trats)]
    dist = np.array([contrastes(df, col, dict(zip(lotes, combo)))
                     for combo in itertools.product(*[perms_l[l] for l in lotes])])
    if bilateral:
        p_frio = np.mean(np.abs(dist[:, 0]) >= abs(obs[0]) - 1e-12)
        p_calor = np.mean(np.abs(dist[:, 1]) >= abs(obs[1]) - 1e-12)
    else:
        p_frio = np.mean(dist[:, 0] <= obs[0] + 1e-12)   # unilateral H1
        p_calor = np.mean(dist[:, 1] >= obs[1] - 1e-12)  # unilateral H2
    p(f"[{rotulo}{' bilateral' if bilateral else ''}] frio−TN = {obs[0]:+.4f} (p = {p_frio:.4f}); calor−TN = {obs[1]:+.4f} (p = {p_calor:.4f}); "
      f"alocações = {len(dist)}")
    return obs, p_frio, p_calor

def misto(col, rotulo):
    d = df.copy(); d["unid"] = d.lote + "_" + d.box.astype(str)
    d["trat"] = pd.Categorical(d.trat, categories=["termoneutro", "frio", "calor"])
    m = smf.mixedlm(f"{col} ~ C(trat) + C(data)", d, groups=d["unid"]).fit(reml=True)
    for k in ["C(trat)[T.frio]", "C(trat)[T.calor]"]:
        ci = m.conf_int().loc[k]
        p(f"[{rotulo} misto] {k}: {m.params[k]:+.4f} (IC95 {ci[0]:+.4f} a {ci[1]:+.4f}; p = {m.pvalues[k]:.4f})")
    p(f"[{rotulo} misto] var(box) = {m.cov_re.iloc[0,0]:.5f}; var(resíduo) = {m.scale:.5f}")

p("=== Ordem por data ===")
rank = []
for (data, idade, lote), g in df.groupby(["data", "idade", "lote"]):
    v = g.set_index("trat").nnd_bl
    rank.append(dict(data=data, idade=idade, lote=lote, frio=v.frio, tn=v.termoneutro, calor=v.calor,
                     frio_menor=v.frio < v.termoneutro and v.frio < v.calor,
                     calor_maior=v.calor > v.termoneutro and v.calor > v.frio))
rk = pd.DataFrame(rank).sort_values("idade"); rk.to_csv(f"{H}/stats_por_data.csv", index=False)
p(rk.round(3).to_string(index=False))
p(f"frio menor: {rk.frio_menor.sum()}/7; calor maior: {rk.calor_maior.sum()}/7")
# frio estritamente abaixo do TN (H1) e calor acima do TN (H2), datas
p(f"frio < TN: {(rk.frio < rk.tn).sum()}/7; calor > TN: {(rk.calor > rk.tn).sum()}/7")

p("\n=== Teste de ordem por lote (planejado) ===")
lotes = sorted(df.lote.unique())
reais = {l: dict(df[df.lote == l].groupby("box").trat.first()) for l in lotes}
pl = [[dict(zip(sorted(reais[l]), pr)) for pr in itertools.permutations([reais[l][b] for b in sorted(reais[l])])] for l in lotes]
obs_o = ordem_lote(df)
do = np.array([ordem_lote(df, dict(zip(lotes, c))) for c in itertools.product(*pl)])
p(df.groupby(["lote", "trat"]).nnd_bl.mean().unstack().round(4).to_string())
p(f"frio menor em {obs_o[0]}/3 lotes (p = {np.mean(do[:,0] >= obs_o[0]):.4f}); "
  f"calor maior em {obs_o[1]}/3 lotes (p = {np.mean(do[:,1] >= obs_o[1]):.4f})")

p("\n=== Aleatorização exata (216 alocações), estatística = diferença média ===")
aleatorizacao("nnd_bl", "NND corpo")
aleatorizacao("nnd", "NND cm corrigida")
aleatorizacao("parede", "fração junto à parede", bilateral=True)
aleatorizacao("disp", "dispersão", bilateral=True)
aleatorizacao("ngrupos", "nº de grupos", bilateral=True)

p("\n=== Modelo misto ===")
misto("nnd_bl", "NND corpo")
misto("nnd", "NND cm")

p("\n=== Diferença relativa média ao TN (NND corpo) ===")
w = df.pivot_table(index="data", columns="trat", values="nnd_bl")
p(f"frio/TN: {(w.frio / w.termoneutro).mean():.3f} (intervalo {(w.frio / w.termoneutro).min():.3f}–{(w.frio / w.termoneutro).max():.3f})")
p(f"calor/TN: {(w.calor / w.termoneutro).mean():.3f} (intervalo {(w.calor / w.termoneutro).min():.3f}–{(w.calor / w.termoneutro).max():.3f})")
p("\nMédias por tratamento (das medianas box-data):")
p(df.groupby("trat")[["nnd_bl", "nnd", "parede", "disp", "ngrupos", "maior_grupo", "ar"]].mean().round(3).to_string())
open(f"{H}/stats_confirmatoria.txt", "w").write("\n".join(out) + "\n")
