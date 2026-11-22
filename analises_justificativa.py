"""Números que justificam as decisões de modelagem citadas no relatório.

1. Qualidade do dataset (ausentes, duplicados, classes).
2. Por que os indicadores macroeconômicos ficam fora da distância.
3. Por que o curso fica fora da distância.
4. Por que a janela do 2º semestre fica fora.
5. Efeito do agrupamento de categorias raras.
Uso: python analises_justificativa.py  ->  resultados/justificativas.txt
"""
import io
import os
import sys
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import grafo_evasao as ge  # noqa: E402


def cv_logistica(X, y, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    f1, auc = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        f1.append(f1_score(y[te], p > .5, average="macro")); auc.append(roc_auc_score(y[te], p))
    return np.mean(f1), np.mean(auc)


def main():
    from sklearn.metrics import normalized_mutual_info_score as NMI
    buf = io.StringIO()
    with redirect_stdout(buf):
        bruto = pd.read_csv("data.csv", sep=";")
        print("== 1. Dataset")
        print(f"linhas: {len(bruto)} | colunas: {bruto.shape[1]} (36 atributos + alvo)")
        print(f"valores ausentes: {int(bruto.isna().sum().sum())} | linhas duplicadas: {int(bruto.duplicated().sum())}")
        print("classes:", bruto.iloc[:, -1].value_counts().to_dict())

        df_all, y_all = ge.carrega("data.csv", incluir_enrolled=True)
        df, y = ge.carrega("data.csv")
        print(f"Dropout x Graduate: {len(y)} alunos, evasão {y.mean():.3f}")

        print("\n== 2. Indicadores macroeconômicos (valores distintos no dataset)")
        for c in ["unemployment", "inflation", "gdp"]:
            print(f"{c}: {df[c].nunique()} valores distintos")
        combos = df[["unemployment", "inflation", "gdp"]].drop_duplicates()
        print(f"combinações distintas dos três: {len(combos)} -> descrevem a coorte/ano de ingresso, não o aluno")

        print("\n== 3. Curso")
        print(f"cursos distintos: {df.course.nunique()} | V de Cramér curso x desfecho: {ge.cramer_v(df.course, y):.3f}")
        cats, nums = ge.JANELAS["1_matricula"]
        N = len(df)
        for incluir in [False, True]:
            cc = cats + (["course"] if incluir else [])
            d2 = df.copy()
            D = ge.dist_gower(d2, cc, nums)
            i, j, w = ge.knn_mutuo(D, 40)
            memb, Q, _ = ge.leiden(N, i, j, w)
            print(f"Gower matrícula {'COM' if incluir else 'SEM'} curso: NMI(comunidade, curso) = {NMI(memb, df.course):.3f} | "
                  f"NMI(comunidade, desfecho) = {NMI(memb, y):.3f} | V comunidade x desfecho = {ge.cramer_v(memb, y):.3f}")

        print("\n== 4. Janela do 2º semestre")
        ev, fo = y == 1, y == 0
        print(f"evadidos com 0 disciplinas inscritas no 2º sem: {(df.s2_enrolled[ev] == 0).mean():.1%} | formados: {(df.s2_enrolled[fo] == 0).mean():.1%}")
        print(f"evadidos com 0 aprovações no 2º sem: {(df.s2_approved[ev] == 0).mean():.1%} | formados: {(df.s2_approved[fo] == 0).mean():.1%}")
        print(f"evadidos com 0 aprovações no 1º sem: {(df.s1_approved[ev] == 0).mean():.1%} | formados: {(df.s1_approved[fo] == 0).mean():.1%}")
        for nome, (c_, n_) in {"matrícula": ge.JANELAS["1_matricula"], "fim do 1º sem": ge.JANELAS["3_fim_1sem"],
                               "fim do 2º sem": ge.JANELA_2SEM}.items():
            X = ge.atributos(df, c_, n_).numpy()
            f1, auc = cv_logistica(X, y)
            print(f"logística, janela {nome}: F1 = {f1:.3f}, AUC = {auc:.3f}")
        q = pd.qcut(df.s2_approved.rank(method="first"), 5, labels=False)
        print(f"V de Cramér, aprovações 2º sem (quintis) x desfecho: {ge.cramer_v(q, y):.3f}")

        print("\n== 5. Categorias raras (< 30 alunos) agrupadas em 'outros'")
        orig = pd.read_csv("data.csv", sep=";"); orig.columns = ge.NOMES
        orig = orig[orig.target != "Enrolled"]
        for c in ge.CAT_FIN:
            vc = orig[c].value_counts(); raras = vc[vc < ge.MIN_CATEGORIA]
            if len(raras):
                print(f"{c}: {len(vc)} categorias -> {len(vc) - len(raras) + 1} | alunos em 'outros': {raras.sum()} ({raras.sum() / len(orig):.1%})")

        print("\n== 6. Referências para k")
        print(f"sqrt(N) = {np.sqrt(N):.0f} | ln(N) = {np.log(N):.1f} | k=40 = {40 / N:.1%} de N")
    txt = buf.getvalue()
    os.makedirs("resultados", exist_ok=True)
    open("resultados/justificativas.txt", "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
