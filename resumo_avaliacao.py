"""Resume a avaliação única: tabela principal, testes estatísticos, sensibilidade a k e figura 14.

Lê resultados/avaliacao_unica/*_divisoes.csv (gerados por avaliacao_unica.py) e grava:
  resultados/avaliacao_unica/resumo.csv, testes.csv, sensibilidade_k.csv, estrutura.csv, tempos.csv
  figuras/14_comparacao_8020.pdf/.png
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PASTA = "resultados/avaliacao_unica"
# só as quatro abordagens em grafo (as referências sem grafo de avaliacao_unica.py ficam fora do resumo)
ORDEM = ["V1 Gower + Leiden", "V3 Associação + Leiden", "V2 Gower + GraphSAGE", "V4 Associação + GraphSAGE"]
JAN = {"1_matricula": "Matrícula", "3_fim_1sem": "Fim do 1º sem.", "4_fim_2sem": "Fim do 2º sem."}


def rotulo(r):
    if r["variação"] != "ref":
        return f"V{r['variação']} {r['distância']} + {r['método']}"
    return "Variável isolada" if r["método"].startswith("Variável isolada") else r["método"]


def carrega():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{PASTA}/*_divisoes.csv")], ignore_index=True)
    d["abordagem"] = d.apply(rotulo, axis=1)
    return d[d.abordagem.isin(ORDEM)].copy()


def testes(d):
    from scipy.stats import friedmanchisquare, wilcoxon
    lin = []
    pares = [("V3 Associação + Leiden", "V1 Gower + Leiden"), ("V4 Associação + GraphSAGE", "V2 Gower + GraphSAGE"),
             ("V2 Gower + GraphSAGE", "V1 Gower + Leiden"), ("V4 Associação + GraphSAGE", "V3 Associação + Leiden")]
    for jan in [j for j in JAN if j in set(d.janela)]:
        for met in ["F1", "AUC"]:
            w = d[(d.janela == jan) & (d.k == 40)].pivot(index="divisão", columns="abordagem", values=met)
            st, p = friedmanchisquare(*[w[a] for a in ORDEM[:4]])
            lin.append({"janela": jan, "métrica": met, "comparação": "Friedman (4 variações)", "diferença média": np.nan,
                        "A melhor em (% divisões)": np.nan, "p": p})
            for a, b in pares:
                diff = w[a] - w[b]
                p = wilcoxon(w[a], w[b]).pvalue if np.any(diff != 0) else 1.0
                lin.append({"janela": jan, "métrica": met, "comparação": f"{a} vs {b}", "diferença média": diff.mean(),
                            "A melhor em (% divisões)": (diff > 0).mean(), "p": p})
    return pd.DataFrame(lin)


def figura(r):
    """Figura 14: desempenho das 4 variações nas três janelas (paleta única de src/cores.py)."""
    import sys
    import matplotlib.pyplot as plt
    sys.path.insert(0, "src")
    import cores as C
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6.8, "pdf.fonttype": 42})
    estilo = {  # (cor, linha, marcador, preenchido)
        "V1 Gower + Leiden": (C.LEIDEN, "-", "o", True), "V3 Associação + Leiden": (C.LEIDEN, "--", "o", False),
        "V2 Gower + GraphSAGE": (C.GRAPHSAGE, "-", "s", True), "V4 Associação + GraphSAGE": (C.GRAPHSAGE, "--", "s", False)}
    jans = [j for j in C.JANELAS if j in set(r.janela)]
    x = np.arange(len(jans))
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.45), layout="constrained")
    for ax, met in zip(axs, ["F1", "AUC"]):
        for ab in ORDEM:
            cor, ls, mk, cheio = estilo[ab]
            s = r[r.abordagem == ab].set_index("janela").loc[jans]
            ax.errorbar(x, s[f"{met} mean"], yerr=s[f"{met} std"], color=cor, ls=ls, lw=1.1, marker=mk, ms=3.6,
                        mfc=cor if cheio else "white", mec=cor, mew=0.9, capsize=0, elinewidth=0.8,
                        label=ab.replace("Associação", "Assoc."), zorder=3 if "V" in ab[:2] else 2)
        ax.set_xticks(x, [C.NOME_JANELA[j].replace("Fim do ", "") for j in jans])
        ax.set_xlim(-0.25, len(jans) - 0.75)
        ax.set_title(met + (" macro" if met == "F1" else ""), loc="left")
        ax.grid(axis="y", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncols=2, frameon=False,
               handlelength=2.6, columnspacing=1.0)
    os.makedirs("figuras", exist_ok=True)
    fig.savefig("figuras/14_comparacao_8020.pdf"); fig.savefig("figuras/14_comparacao_8020.png", dpi=300)
    plt.close(fig)


def main():
    d = carrega()
    principal = d[d.k == 40]
    r = principal.groupby(["janela", "abordagem"])[["F1", "AUC", "revocação evasão"]].agg(["mean", "std"])
    r = r.reset_index()
    r.columns = ["janela", "abordagem"] + [f"{a} {b}" for a, b in r.columns[2:]]
    r["ordem"] = r.abordagem.map({o: i for i, o in enumerate(ORDEM)})
    r = r.sort_values(["janela", "ordem"]).drop(columns="ordem")
    r.round(4).to_csv(f"{PASTA}/resumo.csv", index=False)
    print("=== Comparação única (k=40, 50 divisões 80/20) ===")
    print(r.round(3).to_string(index=False))


    t = testes(d); t.round(4).to_csv(f"{PASTA}/testes.csv", index=False)
    print("\n=== Testes ===\n", t.round(4).to_string(index=False))

    sk = (d.groupby(["janela", "abordagem", "k"])[["F1", "AUC"]].mean().unstack("k").round(3))
    sk.to_csv(f"{PASTA}/sensibilidade_k.csv"); print("\n=== Sensibilidade a k ===\n", sk.to_string())

    est = pd.concat([pd.read_csv(f) for f in glob.glob(f"{PASTA}/*_estrutura.csv")]).sort_values(["janela", "distância", "k"])
    est.round(4).to_csv(f"{PASTA}/estrutura.csv", index=False); print("\n=== Estrutura ===\n", est.round(3).to_string(index=False))

    tem = pd.concat([pd.read_csv(f) for f in glob.glob(f"{PASTA}/*_tempos.csv")])
    tem = tem[tem.k == 40].pivot_table(index="etapa", columns="janela", values="s", aggfunc="mean").round(3)
    tem.to_csv(f"{PASTA}/tempos.csv"); print("\n=== Tempos (s, k=40) ===\n", tem.to_string())

    figura(r)
    print("\nfigura salva em figuras/14_comparacao_8020.pdf")


if __name__ == "__main__":
    main()
