"""Avaliação específica do Leiden e do grafo, nas três janelas (k = 40).

Complementa a avaliação comum 80/20 (avaliacao_unica.py) com as medidas próprias de
um método descritivo, que não fazem sentido para a GraphSAGE:
  - modularidade Q e nº de comunidades com >= 30 alunos;
  - V de Cramér comunidade x desfecho e teste de permutação (1.000 embaralhamentos do
    desfecho com o grafo fixo), nas comunidades com >= 30 alunos;
  - homofilia: fração de arestas que ligam alunos com o mesmo desfecho (e o valor esperado
    ao acaso) e assortatividade pelo desfecho;
  - ARI entre as partições das duas distâncias;
  - melhor variável isolada (V de Cramér), para comparar com a comunidade.
O desfecho só é usado aqui para avaliar; o grafo e o Leiden não o veem.

Uso:  python metricas_leiden.py      -> resultados/avaliacao_unica/leiden_especifica.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import grafo_evasao as ge  # noqa: E402

N_PERM, SEED, K = 1000, 42, 40


def main():
    from sklearn.metrics import adjusted_rand_score
    df, y = ge.carrega("data.csv")
    N = len(df)
    linhas = []
    for jan, (cats, nums) in ge.JANELAS.items():
        membs = {}
        for dist, fn in ge.DISTANCIAS.items():
            D = fn(df, cats, nums)
            i, j, w = ge.knn_mutuo(D, K)
            del D
            memb, Q, g = ge.leiden(N, i, j, w)
            membs[dist] = memb
            grandes = np.isin(memb, np.where(np.bincount(memb) >= 30)[0])
            mm, yy = memb[grandes], y[grandes]
            v = ge.cramer_v(mm, yy)
            rs = np.random.default_rng(SEED)
            nulo = np.array([ge.cramer_v(mm, rs.permutation(yy)) for _ in range(N_PERM)])
            taxa = pd.Series(yy).groupby(mm).mean()
            pi = y.mean()
            g.vs["y"] = y.tolist()
            # melhor variável isolada no conjunto todo (numéricas em quintis)
            melhor, v_melhor = None, -1
            for c in cats + nums:
                gr = df[c].to_numpy() if c in cats else pd.qcut(df[c], 5, labels=False, duplicates="drop").to_numpy()
                vv = ge.cramer_v(gr, y)
                if vv > v_melhor:
                    melhor, v_melhor = c, vv
            linhas.append({
                "janela": jan, "distância": dist, "modularidade": Q,
                "comunidades ≥30": int(len(taxa)), "cobertura": grandes.mean(),
                "evasão mín": taxa.min(), "evasão máx": taxa.max(),
                "V": v, "V nulo máx": nulo.max(), "p": (1 + (nulo >= v).sum()) / (1 + N_PERM),
                "homofilia (mesmo desfecho)": float(np.mean(y[i] == y[j])),
                "homofilia ao acaso": float(pi ** 2 + (1 - pi) ** 2),
                "assortatividade": g.assortativity_nominal("y", directed=False),
                "melhor variável": melhor, "V melhor variável": v_melhor,
            })
            print(jan, dist, f"Q={Q:.3f} V={v:.3f} nulo={nulo.max():.3f}", flush=True)
        ari = adjusted_rand_score(membs["Gower"], membs["Associação"])
        for l in linhas[-2:]:
            l["ARI Gower x Associação"] = ari
    out = pd.DataFrame(linhas)
    os.makedirs("resultados/avaliacao_unica", exist_ok=True)
    out.to_csv("resultados/avaliacao_unica/leiden_especifica.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
