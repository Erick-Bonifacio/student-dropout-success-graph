"""Figuras em alta resolução para os slides (projetor 1920x1080, renderizadas em 2x).

Para cada grafo gera duas imagens com o MESMO layout (comunidades e desfecho), para que o slide
troque uma pela outra no mesmo lugar, e a composição por comunidade em imagem separada.
Gera também a comparação das 4 variações (figura 14) com fontes de slide.

Saída em figuras/slides/.   Uso:  python figuras_slides.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import cores as C  # noqa: E402
import grafo_evasao as ge  # noqa: E402
from figuras_artigo import posicoes  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

PAPEL = "#F5F6F7"      # fundo dos slides
TINTA = "#14202E"
DPI = 200              # 2x: 1 pixel do slide = 2 pixels da imagem
OUT = "figuras/slides"
# comunidades no projetor: só tons saturados (os claros da paleta do artigo somem no fundo claro);
# continuam fora das cores reservadas (vermelho, verde, roxo, âmbar)
COM_SLIDES = ["#1F77B4", "#17BECF", "#8C564B", "#E377C2", "#4D4D4D", "#A6A61C", "#2C3E8C", "#0E8A93", "#B5835A",
              "#D946A8", "#5DA5DA", "#7A5230", "#3B6E8F", "#00A3A3", "#8A8A8A", "#A3536F", "#2E7D9A", "#556270"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18, "axes.edgecolor": "#677180",
                     "axes.labelcolor": TINTA, "xtick.color": TINTA, "ytick.color": TINTA, "text.color": TINTA})


def grafo(df, y, janela, dist, tag):
    N = len(df)
    cats, nums = ge.JANELAS[janela]
    D = ge.DISTANCIAS[dist](df, cats, nums)
    i, j, w = ge.knn_mutuo(D, 40)
    del D
    memb, Q, _ = ge.leiden(N, i, j, w)
    comp = pd.DataFrame({"c": memb, "y": y}).groupby("c").agg(alunos=("y", "size"), evasao=("y", "mean"))
    comp = comp[comp.alunos >= 30].sort_values("evasao", ascending=False)
    comms = list(comp.index)
    cor_c = {c: COM_SLIDES[k % len(COM_SLIDES)] for k, c in enumerate(comms)}
    pos = posicoes(N, i, j, memb, comms)
    ok = ~np.isnan(pos[:, 0])
    e = ok[i] & ok[j]
    lim = np.nanmin(pos, 0), np.nanmax(pos, 0)
    pad = 0.03 * (lim[1] - lim[0])

    def desenha(cores, nome, rotulos):
        # 1680 x 760 px no slide
        fig = plt.figure(figsize=(16.8, 7.6), dpi=DPI, facecolor=PAPEL)
        ax = fig.add_axes([0, 0, 1, 1], facecolor=PAPEL)
        ax.add_collection(LineCollection(np.stack([pos[i[e]], pos[j[e]]], 1), colors=C.ARESTA,
                                         linewidths=0.35, alpha=0.09, zorder=1))
        ax.scatter(pos[ok, 0], pos[ok, 1], c=np.asarray(cores)[ok], s=11, linewidths=0, zorder=2)
        if rotulos:
            for c in comms:
                cx, cy = pos[memb == c].mean(0)
                ax.text(cx, cy, f"C{c}", weight="bold", ha="center", va="center", fontsize=19, zorder=3,
                        path_effects=[pe.withStroke(linewidth=5, foreground="white")])
        ax.set_xlim(lim[0][0] - pad[0], lim[1][0] + pad[0])
        ax.set_ylim(lim[0][1] - pad[1], lim[1][1] + pad[1])
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
        fig.savefig(f"{OUT}/{tag}_{nome}.png", dpi=DPI, facecolor=PAPEL)
        plt.close(fig)

    desenha([cor_c.get(c, "#D5DAE0") for c in memb], "comunidades", True)
    desenha(np.where(y == 1, C.EVADIU, C.FORMOU), "desfecho", False)

    # composição: 1100 x 640 px no slide
    fig, ax = plt.subplots(figsize=(11.0, 6.4), dpi=DPI, facecolor=PAPEL, layout="constrained")
    ax.set_facecolor(PAPEL)
    x = np.arange(len(comp))
    ax.bar(x, comp.evasao, color=C.EVADIU, width=0.78)
    ax.bar(x, 1 - comp.evasao, bottom=comp.evasao, color=C.FORMOU, width=0.78)
    ax.axhline(y.mean(), ls=(0, (2, 2)), c=TINTA, lw=1.6)
    ax.text(len(comp) - 0.45, y.mean() + 0.02, f"evasão geral ({y.mean():.0%})".replace(".", ","), ha="right",
            va="bottom", fontsize=17, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
    fs = 15 if len(comp) <= 13 else 12
    for k, ev in enumerate(comp.evasao):
        if ev >= 0.1:
            ax.text(k, ev - 0.025, f"{ev:.0%}", ha="center", va="top", color="white", fontsize=fs, weight="bold")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(comp) - 0.4)
    ax.set_xticks(x, [f"C{c}\n{n}" for c, n in zip(comp.index, comp.alunos)], fontsize=14 if len(comp) <= 13 else 12)
    for k, c in enumerate(comms):
        ax.add_patch(plt.Rectangle((k - 0.39, -0.055), 0.78, 0.035, color=cor_c[c], clip_on=False))
    ax.tick_params(axis="x", pad=34, length=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("alunos da comunidade")
    fig.savefig(f"{OUT}/{tag}_composicao.png", dpi=DPI, facecolor=PAPEL)
    plt.close(fig)
    print(tag, f"Q={Q:.2f}", "comunidades:", len(comms),
          f"evasão {comp.evasao.min():.0%}-{comp.evasao.max():.0%}",
          "maior risco:", f"C{comms[0]} ({comp.alunos.iloc[0]} alunos, {comp.evasao.iloc[0]:.0%})")


def comparacao():
    r = pd.read_csv("resultados/avaliacao_unica/resumo.csv")
    estilo = {"V1 Gower + Leiden": (C.LEIDEN, "-", "o", True), "V3 Associação + Leiden": (C.LEIDEN, "--", "o", False),
              "V2 Gower + GraphSAGE": (C.GRAPHSAGE, "-", "s", True),
              "V4 Associação + GraphSAGE": (C.GRAPHSAGE, "--", "s", False)}
    jans = C.JANELAS
    x = np.arange(len(jans))
    # 900 x 680 px no slide
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 6.8), dpi=DPI, facecolor=PAPEL, layout="constrained")
    for ax, met in zip(axs, ["F1", "AUC"]):
        ax.set_facecolor(PAPEL)
        for ab, (cor, ls, mk, cheio) in estilo.items():
            s = r[r.abordagem == ab].set_index("janela").loc[jans]
            ax.errorbar(x, s[f"{met} mean"], yerr=s[f"{met} std"], color=cor, ls=ls, lw=2.6, marker=mk, ms=10,
                        mfc=cor if cheio else "white", mec=cor, mew=2.2, elinewidth=1.6,
                        label=ab.replace("Associação", "Assoc."))
        ax.set_xticks(x, ["Matrícula", "1º sem.", "2º sem."], fontsize=15)
        ax.set_xlim(-0.3, len(jans) - 0.7)
        ax.set_title(met + (" macro" if met == "F1" else ""), loc="left", fontsize=20, weight="bold")
        ax.grid(axis="y", color="#D5DAE0", lw=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncols=2, frameon=False,
               fontsize=15, handlelength=3)
    fig.savefig(f"{OUT}/comparacao.png", dpi=DPI, facecolor=PAPEL)
    plt.close(fig)


def demo(df, y):
    """Painel da demonstração: as 6 combinações janela x distância, em comunidades e em desfecho
    (900 x 410 px no slide, renderizadas em 2x), e um JSON com as métricas de cada combinação."""
    import json
    lei = pd.read_csv("resultados/avaliacao_unica/leiden_especifica.csv")
    res = pd.read_csv("resultados/avaliacao_unica/resumo.csv")
    dados = {}
    for jan in C.JANELAS:
        for dist in ["Gower", "Associação"]:
            tag = f"demo_{jan}_{'gower' if dist == 'Gower' else 'assoc'}"
            N = len(df)
            cats, nums = ge.JANELAS[jan]
            D = ge.DISTANCIAS[dist](df, cats, nums)
            i, j, w = ge.knn_mutuo(D, 40)
            del D
            memb, Q, _ = ge.leiden(N, i, j, w)
            comp = pd.DataFrame({"c": memb, "y": y}).groupby("c").agg(alunos=("y", "size"), evasao=("y", "mean"))
            comp = comp[comp.alunos >= 30].sort_values("evasao", ascending=False)
            comms = list(comp.index)
            cor_c = {c: COM_SLIDES[k % len(COM_SLIDES)] for k, c in enumerate(comms)}
            pos = posicoes(N, i, j, memb, comms)
            ok = ~np.isnan(pos[:, 0])
            e = ok[i] & ok[j]
            lim = np.nanmin(pos, 0), np.nanmax(pos, 0)
            pad = 0.03 * (lim[1] - lim[0])
            for nome, cores in [("com", [cor_c.get(c, "#D5DAE0") for c in memb]),
                                ("des", np.where(y == 1, C.EVADIU, C.FORMOU))]:
                fig = plt.figure(figsize=(9.0, 4.1), dpi=DPI, facecolor="white")
                ax = fig.add_axes([0, 0, 1, 1], facecolor="white")
                ax.add_collection(LineCollection(np.stack([pos[i[e]], pos[j[e]]], 1), colors=C.ARESTA,
                                                 linewidths=0.25, alpha=0.08, zorder=1))
                ax.scatter(pos[ok, 0], pos[ok, 1], c=np.asarray(cores)[ok], s=4.5, linewidths=0, zorder=2)
                ax.set_xlim(lim[0][0] - pad[0], lim[1][0] + pad[0])
                ax.set_ylim(lim[0][1] - pad[1], lim[1][1] + pad[1])
                ax.set_aspect("equal", adjustable="datalim")
                ax.axis("off")
                fig.savefig(f"{OUT}/{tag}_{nome}.png", dpi=DPI, facecolor="white")
                plt.close(fig)
            m = lei[(lei.janela == jan) & (lei["distância"] == dist)].iloc[0]
            r = res[res.janela == jan].set_index("abordagem")
            vl = f"V{'1' if dist == 'Gower' else '3'} {dist} + Leiden"
            vs = f"V{'2' if dist == 'Gower' else '4'} {dist} + GraphSAGE"
            dados[tag] = {"Q": round(float(m["modularidade"]), 2), "com": int(m["comunidades ≥30"]),
                          "emin": round(float(m["evasão mín"]) * 100), "emax": round(float(m["evasão máx"]) * 100),
                          "V": round(float(m["V"]), 2), "Vnulo": round(float(m["V nulo máx"]), 2),
                          "hom": round(float(m["homofilia (mesmo desfecho)"]) * 100),
                          "f1L": round(float(r.loc[vl, "F1 mean"]), 3), "f1S": round(float(r.loc[vs, "F1 mean"]), 3),
                          "aucL": round(float(r.loc[vl, "AUC mean"]), 3), "aucS": round(float(r.loc[vs, "AUC mean"]), 3),
                          "arestas": int(len(i))}
            print(tag, dados[tag])
    with open(f"{OUT}/demo.json", "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    df, y = ge.carrega("data.csv")
    if "--demo" in sys.argv:
        demo(df, y)
        return
    grafo(df, y, "1_matricula", "Gower", "matricula_gower")
    grafo(df, y, "3_fim_1sem", "Associação", "1sem_associacao")
    comparacao()
    demo(df, y)


if __name__ == "__main__":
    main()
