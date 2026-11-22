"""Figura 15 do artigo: grafo da matrícula (Gower) com a paleta única de src/cores.py.

(a) comunidades do Leiden (cada comunidade com a sua cor, igual em todos os painéis);
(b) os mesmos alunos coloridos pelo desfecho real (que não entrou no grafo);
(c) composição de cada comunidade (evadiu/formou), com a marca de cor da comunidade.
O layout é o mesmo dos notebooks (bolhas por comunidade, alunos dentro da bolha).

Uso:  python figuras_artigo.py [--janela 1_matricula] [--distancia Gower]
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import cores as C  # noqa: E402
import grafo_evasao as ge  # noqa: E402

SEED = 42


def posicoes(N, i, j, memb, comms):
    import igraph as ig
    idx = {c: k for k, c in enumerate(comms)}
    nC = len(comms)
    tam = np.array([(memb == c).sum() for c in comms])
    random.seed(SEED)
    W = np.zeros((nC, nC))
    m = np.isin(memb[i], comms) & np.isin(memb[j], comms) & (memb[i] != memb[j])
    np.add.at(W, (np.array([idx[c] for c in memb[i][m]], int), np.array([idx[c] for c in memb[j][m]], int)), 1)
    W = W + W.T
    gc = ig.Graph.Weighted_Adjacency(W.tolist(), mode="undirected", loops=False)
    cen = np.array(gc.layout_fruchterman_reingold(weights="weight" if gc.ecount() else None).coords)
    cen = (cen - cen.mean(0)) / (np.abs(cen - cen.mean(0)).max() + 1e-9)
    raio = 0.22 * np.sqrt(tam / tam.max())
    for _ in range(500):
        mexeu = False
        for a in range(nC):
            for b in range(a + 1, nC):
                d = cen[b] - cen[a]
                dab = np.hypot(*d) + 1e-9
                minimo = raio[a] + raio[b] + 0.04
                if dab < minimo:
                    e = (minimo - dab) / 2 * d / dab
                    cen[a] -= e
                    cen[b] += e
                    mexeu = True
        if not mexeu:
            break
    pos = np.full((N, 2), np.nan)
    for k, c in enumerate(comms):
        nos = np.where(memb == c)[0]
        loc = {v: t for t, v in enumerate(nos)}
        mm = (memb[i] == c) & (memb[j] == c)
        g = ig.Graph(n=len(nos), edges=[(loc[a], loc[b]) for a, b in zip(i[mm], j[mm])])
        p = np.array(g.layout_fruchterman_reingold(niter=300).coords)
        p -= np.median(p, 0)
        p = p / (np.percentile(np.hypot(p[:, 0], p[:, 1]), 95) + 1e-9)
        nr = np.hypot(p[:, 0], p[:, 1])
        p[nr > 1] /= nr[nr > 1, None]
        pos[nos] = cen[k] + p * raio[k]
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--janela", default="1_matricula")
    ap.add_argument("--distancia", default="Gower")
    a = ap.parse_args()

    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42})

    df, y = ge.carrega("data.csv")
    N = len(df)
    cats, nums = ge.JANELAS[a.janela]
    D = ge.DISTANCIAS[a.distancia](df, cats, nums)
    i, j, w = ge.knn_mutuo(D, 40)
    del D
    memb, Q, _ = ge.leiden(N, i, j, w)
    comp = (pd.DataFrame({"c": memb, "y": y}).groupby("c").agg(alunos=("y", "size"), evasao=("y", "mean")))
    comp = comp[comp.alunos >= 30].sort_values("evasao", ascending=False)
    comms = list(comp.index)
    # cor fixa por comunidade (ordem do maior para o menor risco)
    cor_c = {c: C.COMUNIDADES[k % len(C.COMUNIDADES)] for k, c in enumerate(comms)}
    pos = posicoes(N, i, j, memb, comms)
    ok = ~np.isnan(pos[:, 0])
    e = ok[i] & ok[j]

    fig = plt.figure(figsize=(7.0, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], left=0.005, right=0.995, bottom=0.21, top=0.9, wspace=0.08)
    axa, axb, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    def desenha(ax, cores, titulo):
        ax.add_collection(LineCollection(np.stack([pos[i[e]], pos[j[e]]], 1), colors=C.ARESTA, linewidths=0.15,
                                         alpha=0.06, zorder=1, rasterized=True))
        ax.scatter(pos[ok, 0], pos[ok, 1], c=np.asarray(cores)[ok], s=1.6, linewidths=0, zorder=2, rasterized=True)
        ax.set_title(titulo, loc="left")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.autoscale_view()

    cores_a = np.array([cor_c.get(c, "#dddddd") for c in memb])
    desenha(axa, cores_a, "(a) Comunidades do Leiden")
    for c in comms:
        cx, cy = pos[memb == c].mean(0)
        axa.text(cx, cy, f"C{c}", weight="bold", ha="center", va="center", fontsize=6, zorder=3,
                 path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    desenha(axb, np.where(y == 1, C.EVADIU, C.FORMOU), "(b) Desfecho real (fora do grafo)")
    axb.legend(handles=[Patch(color=C.EVADIU, label="evadiu"), Patch(color=C.FORMOU, label="formou")],
               loc="lower center", bbox_to_anchor=(0.5, -0.2), ncols=2, frameon=False)

    x = np.arange(len(comp))
    axc.bar(x, comp.evasao, color=C.EVADIU, width=0.78)
    axc.bar(x, 1 - comp.evasao, bottom=comp.evasao, color=C.FORMOU, width=0.78)
    axc.axhline(y.mean(), ls=":", c="k", lw=0.9)
    axc.set_ylim(0, 1)
    axc.set_xlim(-0.6, len(comp) - 0.4)
    axc.set_xticks(x, [f"C{c}\n{n}" for c, n in zip(comp.index, comp.alunos)])
    for t, c in zip(axc.get_xticklabels(), comms):
        t.set_color("black")
    for k, c in enumerate(comms):   # marca de cor da comunidade, igual à do painel (a)
        axc.add_patch(plt.Rectangle((k - 0.39, -0.055), 0.78, 0.04, color=cor_c[c], clip_on=False,
                                    transform=axc.transData))
    axc.tick_params(axis="x", pad=10, length=0)
    axc.yaxis.set_major_formatter(PercentFormatter(1))
    axc.set_title("(c) Composição por comunidade", loc="left")
    axc.spines[["top", "right"]].set_visible(False)
    axc.text(len(comp) - 0.45, y.mean() + 0.015, f"evasão geral ({y.mean():.0%})", ha="right", va="bottom",
             fontsize=6, path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    nome = f"figuras/15_grafo_{a.janela}_{'gower' if a.distancia == 'Gower' else 'associacao'}"
    fig.savefig(nome + ".pdf", dpi=300)
    fig.savefig(nome + ".png", dpi=300)
    print(comp.assign(cor=[cor_c[c] for c in comms]).to_string(), "\nQ =", round(Q, 3), "->", nome)


if __name__ == "__main__":
    main()
