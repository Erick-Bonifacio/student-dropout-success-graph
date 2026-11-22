"""Núcleo do protótipo: dados, janelas, distâncias, grafo kNN mútuo, Leiden e GraphSAGE.

Mesmo pipeline dos notebooks (quatro_variacoes.ipynb), reunido em um módulo para que
todos os scripts de avaliação usem exatamente as mesmas funções.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- dados
NOMES = ["marital", "app_mode", "app_order", "course", "shift", "prev_qual", "prev_grade", "nationality",
         "mother_qual", "father_qual", "mother_occ", "father_occ", "admission_grade", "displaced",
         "special_needs", "debtor", "tuition_ok", "gender", "scholarship", "age", "international",
         "s1_credited", "s1_enrolled", "s1_evaluations", "s1_approved", "s1_grade", "s1_without_eval",
         "s2_credited", "s2_enrolled", "s2_evaluations", "s2_approved", "s2_grade", "s2_without_eval",
         "unemployment", "inflation", "gdp", "target"]

CAT_BASE = ["marital", "app_mode", "shift", "prev_qual", "nationality", "mother_qual", "father_qual",
            "mother_occ", "father_occ", "displaced", "special_needs", "gender", "international"]
NUM_BASE = ["app_order", "prev_grade", "admission_grade", "age"]
CAT_FIN = CAT_BASE + ["debtor", "tuition_ok", "scholarship"]
NUM_S1 = ["s1_credited", "s1_enrolled", "s1_evaluations", "s1_approved", "s1_grade", "s1_without_eval"]
NUM_S2 = [c.replace("s1_", "s2_") for c in NUM_S1]

# janelas de tempo: (categóricas, numéricas) conhecidas em cada momento
JANELAS = {
    "1_matricula": (CAT_BASE, NUM_BASE),
    "3_fim_1sem": (CAT_FIN, NUM_BASE + NUM_S1),
    "4_fim_2sem": (CAT_FIN, NUM_BASE + NUM_S1 + NUM_S2),
}
JANELA_2SEM = JANELAS["4_fim_2sem"]
MIN_CATEGORIA = 30                                     # categorias com menos alunos viram "outros" (-1)


def carrega(caminho: str, incluir_enrolled: bool = False):
    """Lê o CSV do UCI 697. Retorna (df de atributos, y) com y = 1 evadiu, 0 formou (-1 Enrolled)."""
    df = pd.read_csv(caminho, sep=";")
    df.columns = NOMES
    if not incluir_enrolled:
        df = df[df.target != "Enrolled"].reset_index(drop=True)
    y = df.pop("target").map({"Dropout": 1, "Graduate": 0, "Enrolled": -1}).to_numpy()
    for c in CAT_FIN:
        vc = df[c].value_counts()
        df[c] = df[c].where(df[c].isin(vc[vc >= MIN_CATEGORIA].index), -1)
    return df, y


# ----------------------------------------------------------------------------- distâncias
def dist_gower(df, cats, nums):
    import gower
    X = pd.concat([df[cats].astype(str), df[nums].astype(float)], axis=1)
    return gower.gower_matrix(X, cat_features=[True] * len(cats) + [False] * len(nums))


def dist_associacao(df, cats, nums):
    """Preset u_dep do manydist (van de Velden et al., 2026), reimplementado em Python."""
    N = len(df)
    D = np.zeros((N, N))
    codes = [pd.Categorical(df[c]).codes for c in cats]
    Q = [c.max() + 1 for c in codes]
    pos = np.cumsum([0] + Q)
    m = len(cats)
    Z = np.zeros((N, sum(Q)))
    for j, c in enumerate(codes):
        Z[np.arange(N), pos[j] + c] = 1
    ZZ = Z.T @ Z
    np.fill_diagonal(ZZ, 0)
    P = ZZ / np.maximum(Z.sum(0)[:, None], 1)                 # perfil de cada categoria nas demais variáveis
    for j, c in enumerate(codes):
        Pj = P[pos[j]:pos[j + 1]]
        delta = np.abs(Pj[:, None, :] - Pj[None, :, :]).sum(-1) / (2 * (m - 1))   # variação total média
        f = np.bincount(c, minlength=Q[j]) / N
        media = f @ delta @ f
        if media > 0:
            D += delta[c[:, None], c[None, :]] / media          # comensurabilidade
    X = df[nums].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    X = X @ np.linalg.svd(X, full_matrices=False)[2].T          # rotação por PCA
    for k in range(X.shape[1]):
        dk = np.abs(X[:, k, None] - X[None, :, k])
        D += dk / dk.mean()
    return D.astype(np.float32)


DISTANCIAS = {"Gower": dist_gower, "Associação": dist_associacao}


# ----------------------------------------------------------------------------- grafo
def knn_mutuo(D, k: int = 40):
    """Aresta {i,j} se i está entre os k mais próximos de j E j entre os k mais próximos de i."""
    N = D.shape[0]
    Dm = np.round(D, 6)
    np.fill_diagonal(Dm, np.inf)
    viz = np.argsort(Dm, axis=1, kind="stable")[:, :k]       # empate desfeito pelo índice
    A = np.zeros((N, N), bool)
    A[np.repeat(np.arange(N), k), viz.ravel()] = True
    i, j = np.where(np.triu(A & A.T, 1))
    return i, j, np.maximum(1 - D[i, j] / D.max(), 1e-3)     # peso = semelhança em (0, 1]


def leiden(N, i, j, w, seed: int = 0):
    import igraph as ig
    import leidenalg as la
    g = ig.Graph(n=N, edges=list(zip(i, j)))
    g.es["weight"] = w
    part = la.find_partition(g, la.ModularityVertexPartition, weights="weight", seed=seed)
    return np.array(part.membership), part.modularity, g


def cramer_v(a, b):
    from scipy.stats import chi2_contingency
    T = pd.crosstab(a, b).values
    if min(T.shape) < 2:
        return 0.0
    return float(np.sqrt(chi2_contingency(T, correction=False)[0] / (T.sum() * (min(T.shape) - 1))))


# ----------------------------------------------------------------------------- previsores por grupo
def taxa_por_grupo(grupo, y, treino, a: float = 5.0):
    """P(evadir) = taxa do grupo no treino, suavizada para a taxa geral (a alunos 'médios')."""
    pi = y[treino].mean()
    s = pd.Series(y[treino]).groupby(grupo[treino]).agg(["sum", "count"])
    taxa = (s["sum"] + a * pi) / (s["count"] + a)
    return pd.Series(grupo).map(taxa).fillna(pi).to_numpy(), pi


def melhor_variavel(df, cats, nums, y, treino):
    """Escolhe, só no treino, a variável isolada com maior V de Cramér (numéricas em quintis do treino)."""
    melhor, grupo_melhor, v_melhor = None, None, -1.0
    for c in cats + nums:
        if c in cats:
            g = df[c].to_numpy()
        else:
            cortes = np.unique(np.quantile(df[c].to_numpy()[treino], [0.2, 0.4, 0.6, 0.8]))
            g = np.digitize(df[c].to_numpy(), cortes)
        v = cramer_v(g[treino], y[treino])
        if v > v_melhor:
            melhor, grupo_melhor, v_melhor = c, g, v
    return melhor, grupo_melhor


# ----------------------------------------------------------------------------- redes neurais
def atributos(df, cats, nums):
    import torch
    oh = pd.get_dummies(df[cats].astype(str), dtype=np.float32).values
    num = df[nums].to_numpy(np.float32)
    num = (num - num.mean(0)) / num.std(0)
    return torch.tensor(np.hstack([oh, num]))


def treina(model, X, ei, y, tr, va, epocas=300, paciencia=30, lr=0.01):
    """Adam, pesos de classe, parada antecipada pelo F1 macro na validação (retirada do treino)."""
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score
    yt = torch.tensor(y)
    peso = torch.tensor(len(tr) / (2 * np.bincount(y[tr])), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    melhor, estado, sem = -1.0, None, 0
    for _ in range(epocas):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(X, ei)[tr], yt[tr], weight=peso).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            f = f1_score(y[va], model(X, ei)[va].argmax(1), average="macro")
        if f > melhor:
            melhor, estado, sem = f, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            sem += 1
            if sem >= paciencia:
                break
    model.load_state_dict(estado)
    model.eval()
    with torch.no_grad():
        return F.softmax(model(X, ei), 1)[:, 1].numpy()


def graphsage(d):
    from torch_geometric.nn.models import GraphSAGE
    return GraphSAGE(d, 64, num_layers=2, out_channels=2, dropout=0.5)


def mlp_sem_arestas(d):
    import torch
    from torch_geometric.nn.models import MLP

    class SemArestas(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = MLP(in_channels=d, hidden_channels=64, out_channels=2, num_layers=2, dropout=0.5, norm=None)

        def forward(self, x, ei):
            return self.m(x)

    return SemArestas()
