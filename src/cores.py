"""Paleta única das figuras do artigo: cada conceito tem sempre a mesma cor.

    desfecho     evadiu = vermelho, formou = verde (também nos mapas de risco: verde -> vermelho)
    método       Leiden = roxo, GraphSAGE = âmbar; referências sem grafo = tons de cinza
    distância    Gower = linha cheia / marcador cheio; Associação = linha tracejada / marcador vazado
    janela       sempre no eixo ou no painel, na ordem matrícula -> 1º semestre -> 2º semestre
    comunidade   cores categóricas que evitam as cores reservadas acima (vermelho, verde, roxo, âmbar)
"""
EVADIU = "#C0392B"
FORMOU = "#27AE60"

LEIDEN = "#534AB7"
GRAPHSAGE = "#E08A00"
VAR_ISOLADA = "#A8A6A0"
LOGISTICA = "#4A4945"
MLP = "#7D7B75"

COR_METODO = {"Leiden": LEIDEN, "GraphSAGE": GRAPHSAGE, "Variável isolada": VAR_ISOLADA,
              "Logística": LOGISTICA, "MLP (sem arestas)": MLP}
LINHA_DIST = {"Gower": "-", "Associação": "--"}

JANELAS = ["1_matricula", "3_fim_1sem", "4_fim_2sem"]
NOME_JANELA = {"1_matricula": "Matrícula", "3_fim_1sem": "Fim do 1º sem.", "4_fim_2sem": "Fim do 2º sem."}

# comunidades: azuis, cianos, marrons, rosas, oliva e cinzas (nenhuma cor reservada)
COMUNIDADES = ["#1F77B4", "#17BECF", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#393B79", "#9EDAE5",
               "#C49C94", "#F7B6D2", "#AEC7E8", "#636363", "#6B8E9F", "#D4A373", "#BDBDBD", "#2F4B7C",
               "#A0522D", "#5F9EA0"]

ARESTA = "#999999"
