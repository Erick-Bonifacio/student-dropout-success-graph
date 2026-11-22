"""Avaliação única das 4 variações (protocolo 80/20).

Todas as variações e referências são avaliadas nas MESMAS divisões treino/teste:
validação cruzada estratificada de 5 dobras (cada dobra = 80% treino, 20% teste),
repetida R vezes com sorteios diferentes. Métricas iguais para todos: F1 macro, AUC e
revocação da evasão. O desfecho nunca entra na construção do grafo nem do Leiden; os
rótulos dos 20% de teste nunca entram no treino.

    variação 1: Gower      + Leiden      variação 2: Gower      + GraphSAGE
    variação 3: Associação + Leiden      variação 4: Associação + GraphSAGE
    referências (sem grafo): melhor variável isolada, regressão logística, MLP

Uso:
    python avaliacao_unica.py --janela 1_matricula --k 40 --repeticoes 10
Saída em resultados/avaliacao_unica/ (um CSV por janela e k).
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import grafo_evasao as ge  # noqa: E402


def metricas(y, p, limiar):
    from sklearn.metrics import f1_score, recall_score, roc_auc_score
    pred = (p > limiar).astype(int)
    return {"F1": f1_score(y, pred, average="macro"), "AUC": roc_auc_score(y, p),
            "revocação evasão": recall_score(y, pred)}


def estrutura(N, i, j):
    import igraph as ig
    g = ig.Graph(n=N, edges=list(zip(i, j)))
    grau = np.array(g.degree())
    comps = g.connected_components()
    gc = comps.giant()
    return {"arestas": g.ecount(), "densidade": g.ecount() / (N * (N - 1) / 2), "grau médio": grau.mean(),
            "isolados": int((grau == 0).sum()), "componentes": len(comps),
            "maior componente (%)": max(comps.sizes()) / N,
            "agrupamento (transitividade)": g.transitivity_undirected(),
            "caminho médio": gc.average_path_length(), "diâmetro": gc.diameter()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="data.csv")
    ap.add_argument("--janela", default="1_matricula", choices=list(ge.JANELAS))
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--repeticoes", type=int, default=10)
    ap.add_argument("--sem-gnn", action="store_true", help="pula GraphSAGE/MLP (teste rápido)")
    ap.add_argument("--saida", default="resultados/avaliacao_unica")
    a = ap.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    os.makedirs(a.saida, exist_ok=True)
    df, y = ge.carrega(a.dados)
    N = len(df)
    cats, nums = ge.JANELAS[a.janela]
    X = ge.atributos(df, cats, nums)
    divisoes = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeticoes, random_state=42).split(np.zeros(N), y))
    print(f"{a.janela} k={a.k}: {N} alunos, {len(divisoes)} divisões 80/20", flush=True)

    # grafo e Leiden: uma vez por distância (não usam o desfecho)
    grafos, linhas_estr, tempos = {}, [], []
    for dist, fn in ge.DISTANCIAS.items():
        t0 = time.perf_counter(); D = fn(df, cats, nums); t_d = time.perf_counter() - t0
        t0 = time.perf_counter(); i, j, w = ge.knn_mutuo(D, a.k); t_k = time.perf_counter() - t0
        t0 = time.perf_counter(); memb, Q, _ = ge.leiden(N, i, j, w); t_l = time.perf_counter() - t0
        ei = torch.tensor(np.vstack([np.r_[i, j], np.r_[j, i]]), dtype=torch.long)
        grafos[dist] = dict(memb=memb, ei=ei)
        n_com = int((np.bincount(memb) >= 30).sum())
        linhas_estr.append({"janela": a.janela, "k": a.k, "distância": dist, "modularidade": Q,
                            "comunidades ≥30": n_com, **estrutura(N, i, j)})
        tempos += [{"etapa": f"distância {dist}", "s": t_d}, {"etapa": "kNN mútuo", "s": t_k}, {"etapa": "Leiden", "s": t_l}]
        del D

    res = []
    t_metodo = {}

    def registra(metodo, variacao, dist, s, m):
        res.append({"janela": a.janela, "k": a.k, "método": metodo, "variação": variacao, "distância": dist,
                    "divisão": s, **m})

    for s, (tr_all, te) in enumerate(divisoes):
        tr, va = train_test_split(tr_all, test_size=0.15, stratify=y[tr_all], random_state=s)

        # variações 1 e 3: Leiden -> taxa de evasão da comunidade calculada só no treino
        for dist, var in [("Gower", "1"), ("Associação", "3")]:
            t0 = time.perf_counter()
            p, pi = ge.taxa_por_grupo(grafos[dist]["memb"], y, tr_all)
            t_metodo[f"V{var} Leiden"] = t_metodo.get(f"V{var} Leiden", 0) + time.perf_counter() - t0
            registra("Leiden", var, dist, s, metricas(y[te], p[te], pi))

        # referência: melhor variável isolada (escolhida só no treino), mesmo previsor por grupo
        t0 = time.perf_counter()
        var_nome, grupo = ge.melhor_variavel(df, cats, nums, y, tr_all)
        p, pi = ge.taxa_por_grupo(grupo, y, tr_all)
        t_metodo["ref. variável isolada"] = t_metodo.get("ref. variável isolada", 0) + time.perf_counter() - t0
        registra(f"Variável isolada ({var_nome})", "ref", "-", s, metricas(y[te], p[te], pi))

        # referência: regressão logística (sem grafo), pesos de classe
        t0 = time.perf_counter()
        lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X.numpy()[tr_all], y[tr_all])
        p = lr.predict_proba(X.numpy())[:, 1]
        t_metodo["ref. logística"] = t_metodo.get("ref. logística", 0) + time.perf_counter() - t0
        registra("Logística", "ref", "-", s, metricas(y[te], p[te], 0.5))

        if a.sem_gnn:
            continue
        # variações 2 e 4: GraphSAGE no grafo de cada distância
        for dist, var in [("Gower", "2"), ("Associação", "4")]:
            torch.manual_seed(s)
            t0 = time.perf_counter()
            p = ge.treina(ge.graphsage(X.shape[1]), X, grafos[dist]["ei"], y, tr, va)
            t_metodo[f"V{var} GraphSAGE"] = t_metodo.get(f"V{var} GraphSAGE", 0) + time.perf_counter() - t0
            registra("GraphSAGE", var, dist, s, metricas(y[te], p[te], 0.5))
        # referência: MLP de mesma arquitetura, sem arestas
        torch.manual_seed(s)
        t0 = time.perf_counter()
        p = ge.treina(ge.mlp_sem_arestas(X.shape[1]), X, grafos["Gower"]["ei"], y, tr, va)
        t_metodo["ref. MLP"] = t_metodo.get("ref. MLP", 0) + time.perf_counter() - t0
        registra("MLP (sem arestas)", "ref", "-", s, metricas(y[te], p[te], 0.5))
        if s % 5 == 4:
            print(f"  repetição {s // 5 + 1}/{a.repeticoes} ok", flush=True)

    base = f"{a.saida}/{a.janela}_k{a.k}"
    pd.DataFrame(res).to_csv(f"{base}_divisoes.csv", index=False)
    pd.DataFrame(linhas_estr).to_csv(f"{base}_estrutura.csv", index=False)
    tempos += [{"etapa": f"{m} (média por divisão)", "s": v / len(divisoes)} for m, v in t_metodo.items()]
    pd.DataFrame(tempos).assign(janela=a.janela, k=a.k).to_csv(f"{base}_tempos.csv", index=False)
    r = pd.DataFrame(res)
    r["método"] = r["método"].str.replace(r"Variável isolada \(.*\)", "Variável isolada", regex=True)
    print(r.groupby(["variação", "método", "distância"])[["F1", "AUC", "revocação evasão"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
