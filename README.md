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

Usamos só **Dropout × Graduate** (3.630 alunos, 39% de evasão). *Enrolled* fica fora: não é desfecho final, o aluno ainda está no curso.

## Como rodar

```bash
pip install -r requirements.txt jupyter
jupyter notebook quatro_variacoes.ipynb
```

`data.csv` precisa estar na mesma pasta do notebook. Roda inteiro em poucos minutos num CPU, sem cache. Os números vão para `resultados/` e as figuras para `figuras/`.

## Resultados

| Janela | Distância | Leiden: V de Cramér (comunidades) | F1 GraphSAGE | F1 MLP (sem arestas) |
|--------|-----------|-----------------------------------|--------------|----------------------|
| 1_matricula | Gower      | 0,32 (12) | 0,68 | 0,68 |
| 1_matricula | Associação | 0,33 (16) | 0,67 | 0,68 |
| 3_fim_1sem  | Gower      | 0,46 (11) | 0,86 | 0,86 |
| 3_fim_1sem  | Associação | 0,58 (18) | 0,86 | 0,87 |

p = 0,001 em todos os testes do Leiden (o menor possível com 1.000 permutações). Comunidades com pelo menos 30 alunos. F1 macro, média das 5 dobras.

Leiden é reprodutível exatamente; os F1 da GNN variam cerca de 0,01 entre máquinas/versões do torch; a conclusão (GNN ≈ MLP) não depende disso.

Números completos em `resultados/leiden_resumo.csv` e `resultados/gnn_resumo.csv`.

## Figuras

Em `figuras/`, cada uma em PDF (para o LaTeX, largura 6,3 pol., fonte 9 pt) e PNG 300 dpi. `{janela}` = `1_matricula` ou `3_fim_1sem`; `{distancia}` = `gower` ou `associacao`.

- `01_grafo_comunidades_{janela}_{distancia}`: (a) comunidades do Leiden em bolhas, com nº de alunos; (b) o mesmo grafo pintado por evadiu/formou.
- `02_mapa_risco_{janela}_{distancia}`: cada aluno com a cor da taxa de evasão da sua comunidade.
- `03_composicao_{janela}`: fração de evadidos/formados por comunidade, (a) Gower e (b) Associação.
- `04_perfil_{janela}_{distancia}`: heatmap das variáveis por comunidade, em desvios-padrão da média geral.
- `05_gower_vs_associacao_{janela}`: para onde vai cada comunidade do Gower na partição da Associação, com o ARI.
- `06_gnn_f1`: F1 da GraphSAGE, do MLP sem arestas e da logística, nas 4 combinações.
- `07_gnn_grafo_{janela}_{distancia}`: (a) risco previsto pela GNN fora da dobra; (b) erros da GNN, nas posições da figura 01.

## Principais achados

- **Os perfis existem nas duas distâncias.** As comunidades separam a evasão muito além do acaso, e mais ainda depois do 1º semestre (V de 0,32–0,33 na matrícula, 0,46–0,58 no fim do 1º semestre).
- **O risco na matrícula não depende da distância.** As duas apontam o mesmo perfil: alunos mais velhos com pais de escolaridade desconhecida, com cerca de 70% de evasão (Gower: 142 alunos, 70%; Associação: 166, 67%), contra 39% no geral. Ingressantes "maiores de 23" ficam perto de 60%.
- **No fim do 1º semestre aparecem os grupos extremos:** notas baixas, a maioria (65%) com mensalidade atrasada (Gower: 395 alunos, 83% de evasão); a associação isola 267 alunos com 96% (notas 1,5 dp abaixo da média).
- **A vizinhança não ajuda a prever.** A GraphSAGE empata com o mesmo modelo sem arestas, porque as arestas saem dos mesmos atributos que o modelo já recebe. O grafo serve para descrever perfis, não para prever melhor.
- Associação não é causa: os perfis servem para oferecer apoio, nunca para rotular alunos.
