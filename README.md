# Evasão estudantil em grafos — 4 variações

Projeto de Grafos (UNIFEI, SMAC03, Cenário 3). Dataset UCI 697 (*Predict Students' Dropout and Academic Success*).

Cada aluno é um vértice; a aresta existe quando dois alunos estão entre os 40 mais parecidos um do outro (kNN mútuo). O desfecho nunca entra no grafo. Entre as variações muda **só a distância** (linha) e **o método** (coluna); dados, janelas, k, grafo e avaliação são idênticos.

|                | **Leiden** — que perfis existem? | **GNN GraphSAGE** — a vizinhança ajuda a prever? |
|----------------|----------------------------------|--------------------------------------------------|
| **Gower**      | variação 1                       | variação 2                                       |
| **Associação** | variação 3                       | variação 4                                       |

- **Leiden:** encontra comunidades sem olhar o desfecho e mede, com o V de Cramér e um teste de permutação (1.000 embaralhamentos), se as comunidades têm taxas de evasão diferentes.
- **GNN:** compara GraphSAGE com um MLP de mesma arquitetura sem arestas (validação cruzada com 5 dobras). A diferença entre os dois diz se a vizinhança traz informação nova.
- **Associação** é a *distância* por associação (preset `u_dep` do pacote R `manydist`), não regras de associação/Apriori.

Duas janelas de tempo: `1_matricula` (só o que se sabe na matrícula) e `3_fim_1sem` (inclui financeiro e notas do 1º semestre).

## Duas classes

Usamos só **Dropout × Graduate** (3.630 alunos). *Enrolled* fica fora: não é desfecho final, o aluno ainda está no curso.

## Como rodar

```bash
pip install -r requirements.txt
jupyter notebook quatro_variacoes.ipynb
```

`data.csv` precisa estar na mesma pasta do notebook. Roda inteiro em poucos minutos num CPU, sem cache. As saídas vão para `resultados/`.

## Resultados

| Janela | Distância | Leiden: V de Cramér (comunidades) | F1 GraphSAGE | F1 MLP (sem arestas) |
|--------|-----------|-----------------------------------|--------------|----------------------|
| 1_matricula | Gower      | 0,32 (12) | 0,69 | 0,68 |
| 1_matricula | Associação | 0,33 (16) | 0,67 | 0,68 |
| 3_fim_1sem  | Gower      | 0,46 (11) | 0,86 | 0,86 |
| 3_fim_1sem  | Associação | 0,58 (18) | 0,86 | 0,86 |

p = 0,001 em todos os testes do Leiden (o menor possível com 1.000 permutações). Comunidades com pelo menos 30 alunos. F1 macro, média das 5 dobras.

Leiden é reprodutível exatamente; os F1 da GNN variam cerca de 0,01 entre máquinas/versões do torch; a conclusão (GNN ≈ MLP) não depende disso.

Números completos em `resultados/leiden_resumo.csv` e `resultados/gnn_resumo.csv`; composição das comunidades em `resultados/leiden_composicao.png`.
