# Evasão estudantil em grafos — 4 variações

Projeto de Grafos (UNIFEI, SMAC03, Cenário 3). Dataset UCI 697 (*Predict Students' Dropout and Academic Success*).

Cada aluno é um vértice; a aresta existe quando dois alunos estão entre os 40 mais parecidos um do outro (kNN mútuo). O desfecho nunca entra no grafo. Entre as variações muda **só a distância** (linha) e **o método** (coluna); dados, janelas, k, grafo e avaliação são idênticos.

|                | **Leiden** — que perfis existem? | **GNN GraphSAGE** — a vizinhança ajuda a prever? |
|----------------|----------------------------------|--------------------------------------------------|
| **Gower**      | variação 1                       | variação 2                                       |
| **Associação** | variação 3                       | variação 4                                       |

- **Leiden:** encontra comunidades sem olhar o desfecho e mede, com o V de Cramér e um teste de permutação (1.000 embaralhamentos), se as comunidades têm taxas de evasão diferentes.
- **GNN:** a GraphSAGE aprende com os atributos do aluno e a média dos vizinhos (2 camadas, 64 neurônios) e prevê o desfecho; é avaliada nas mesmas divisões 80/20 que o Leiden.
- **Associação** é a *distância* por associação (preset `u_dep` do pacote R `manydist`), não regras de associação/Apriori.

Três janelas da trajetória: `1_matricula` (só o que se sabe na matrícula), `3_fim_1sem` (inclui financeiro e notas do 1º semestre) e `4_fim_2sem` (inclui também o 2º semestre). O desfecho é sempre o alvo.

## Avaliação única (régua comum para as 4 variações)

As quatro variações são comparadas **no mesmo protocolo**: validação cruzada estratificada de 5 dobras
(cada dobra = 80% treino, 20% teste), repetida 10 vezes, nas mesmas 50 divisões para todos os métodos.
Métricas iguais para todos: F1 macro, AUC e revocação da evasão. O Leiden entra na comparação prevendo,
para cada aluno de teste, a taxa de evasão da sua comunidade calculada só com o treino. O artigo compara só as
quatro abordagens em grafo; `avaliacao_unica.py` ainda calcula referências sem grafo (variável isolada, logística, MLP),
mas `resumo_avaliacao.py` as deixa fora da tabela, dos testes e da figura.

```bash
python avaliacao_unica.py --janela 1_matricula --k 40 --repeticoes 10
python avaliacao_unica.py --janela 3_fim_1sem  --k 40 --repeticoes 10
python avaliacao_unica.py --janela 4_fim_2sem  --k 40 --repeticoes 10
python avaliacao_unica.py --janela 1_matricula --k 20 --repeticoes 2   # sensibilidade a k (idem 60 e 3_fim_1sem)
python resumo_avaliacao.py        # tabela, testes estatísticos, sensibilidade, figura 14
python analises_justificativa.py  # números que justificam as decisões de modelagem
python metricas_leiden.py         # avaliação específica do Leiden (Q, V de Cramér, permutação, homofilia), 3 janelas
python figuras_artigo.py          # figura 15 (grafo + composição) com a paleta única de src/cores.py
python figuras_slides.py          # figuras dos slides em alta resolução (figuras/slides/)
python slides/montar_slides.py    # monta slides/apresentacao_grafos.html e slides/demonstracao.html
```

Resultados em `resultados/avaliacao_unica/` e `resultados/justificativas.txt`. O núcleo do protótipo
(dados, distâncias, grafo, Leiden, GraphSAGE) fica em `src/grafo_evasao.py` e é usado pelos scripts.
O relatório final (template LaTeX do SIGAA, pronto para compilar) está em `relatorio_sigaa/`. As cores das figuras
seguem uma paleta única (`src/cores.py`): evadiu = vermelho, formou = verde, Leiden = roxo, GraphSAGE = âmbar;
Gower = linha cheia, associação = tracejada. Os slides da apresentação estão em `slides/apresentacao_grafos.html`.

## Duas classes

Usamos só **Dropout × Graduate** (3.630 alunos, 39% de evasão). *Enrolled* fica fora: não é desfecho final, o aluno ainda está no curso.

## Como rodar

```bash
pip install -r requirements.txt jupyter
jupyter notebook quatro_variacoes.ipynb   # fase 1
jupyter notebook previsao_enrolled.ipynb  # fase 2
jupyter notebook estrutura_relacional.ipynb  # fase 3
```

`data.csv` precisa estar na mesma pasta do notebook. Roda inteiro em poucos minutos num CPU, sem cache. Os números vão para `resultados/` e as figuras para `figuras/`.

## Resultados

**Avaliação única** (média nas mesmas 50 divisões 80/20; `resultados/avaliacao_unica/resumo.csv`):

| Abordagem | F1 / AUC matrícula | F1 / AUC fim 1º sem | F1 / AUC fim 2º sem |
|-----------|--------------------|---------------------|---------------------|
| V1 Gower + Leiden          | 0,631 / 0,679 | 0,701 / 0,763 | 0,762 / 0,833 |
| V3 Associação + Leiden     | 0,634 / 0,687 | 0,740 / 0,827 | 0,760 / 0,849 |
| V2 Gower + GraphSAGE       | 0,675 / 0,730 | 0,863 / 0,925 | 0,894 / 0,946 |
| V4 Associação + GraphSAGE  | 0,674 / 0,733 | 0,865 / 0,923 | 0,893 / 0,946 |

Friedman p < 0,001 nas três janelas. GraphSAGE supera Leiden em 98–100% das divisões; no Leiden, a associação supera o Gower (com clareza no 1º semestre); a GraphSAGE não muda com a distância.

**Estrutura das comunidades** (Leiden, V de Cramér com o desfecho, nº de comunidades):

| Janela | Gower | Associação |
|--------|-------|------------|
| 1_matricula | 0,32 (12) | 0,33 (16) |
| 3_fim_1sem  | 0,46 (11) | 0,58 (18) |
| 4_fim_2sem  | 0,61 (13) | 0,62 (16) |

p = 0,001 em todos os testes do Leiden (o menor possível com 1.000 permutações). Comunidades com pelo menos 30 alunos.

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
- `08_enrolled_calibracao_{janela}`: calibração nos alunos conhecidos (evasão observada × prevista), (a) Leiden e (b) GNN.
- `09_enrolled_distribuicao_{janela}`: distribuição do risco previsto para os 794 *Enrolled*, (a) Leiden e (b) GNN.
- `10_enrolled_grafo_{janela}_{distancia}`: os *Enrolled* no grafo dos 4.424 alunos, coloridos pelo risco previsto pela GNN.
- `11_gemeos_destinos_diferentes`: pares vizinhos na matrícula com desfechos opostos, comparados no que veio depois, (a) Gower e (b) Associação.
- `12_comunidade_vs_variaveis_isoladas`: V de Cramér da comunidade contra as 10 melhores variáveis sozinhas, (a) matrícula e (b) fim do 1º semestre.
- `13_combinacao_de_fatores`: evasão por combinação de idade ≥ 25 e pais com escolaridade desconhecida.

## Principais achados

- **Os perfis existem nas duas distâncias.** As comunidades separam a evasão muito além do acaso, e mais ainda depois do 1º semestre (V de 0,32–0,33 na matrícula, 0,46–0,58 no fim do 1º semestre).
- **O risco na matrícula não depende da distância.** As duas apontam o mesmo perfil: alunos mais velhos com pais de escolaridade desconhecida, com cerca de 70% de evasão (Gower: 142 alunos, 70%; Associação: 166, 67%), contra 39% no geral. Ingressantes "maiores de 23" ficam perto de 60%.
- **No fim do 1º semestre aparecem os grupos extremos:** notas baixas, a maioria (65%) com mensalidade atrasada (Gower: 395 alunos, 83% de evasão); a associação isola 267 alunos com 96% (notas 1,5 dp abaixo da média).
- **GraphSAGE para prever, Leiden para descrever.** A GraphSAGE é a melhor das quatro variações em todas as janelas; o Leiden troca acerto individual por grupos concretos e interpretáveis.
- Associação não é causa: os perfis servem para oferecer apoio, nunca para rotular alunos.

## Fase 2: o futuro dos Enrolled

Em `previsao_enrolled.ipynb`, os 794 *Enrolled* entram no grafo como vértices sem rótulo e recebem uma probabilidade de evadir: no Leiden, a taxa de evasão da sua comunidade; na GNN (e no MLP de referência), o modelo treinado nos 3.630 alunos com desfecho. Somando as probabilidades, são esperadas 295/299 evasões pelo Leiden, 286/280 pela GNN e 292/299 pelo MLP na matrícula (Gower/Associação), e 285/285, 328/324 e 327/335 no fim do 1º semestre, ou seja, cerca de 35% a 42% dos *Enrolled*. A confiança é a calibração medida nos alunos conhecidos (ECE ≤ 0,03 em todos os métodos); vale se o padrão se mantiver.

Números em `resultados/enrolled_resumo.csv` e `resultados/enrolled_calibracao.csv`. O Leiden é exato; a GNN e o MLP variam cerca de 0,01 entre execuções.

## Fase 3: o que a estrutura relacional revela

Em `estrutura_relacional.ipynb` (determinístico, sem GNN): no grafo da matrícula há 15.560 (Gower) e 16.268 (Associação) pares de "gêmeos", alunos vizinhos com desfechos opostos (39% das arestas); eles quase não diferem na matrícula, mas o que evadiu aprova em média 2,7 disciplinas no 1º semestre contra 6,2 do que se formou, e 29% têm mensalidade atrasada contra 1%. Os desfechos se agrupam no grafo sem que ele os tenha visto: 0,61/0,61 das arestas ligam alunos com o mesmo desfecho na matrícula e 0,73/0,78 no fim do 1º semestre (Gower/Associação), contra 0,52 ao acaso. Na separação do desfecho, a comunidade **não** supera a melhor variável sozinha (idade, V = 0,33, contra 0,32/0,33 na matrícula; aprovações no 1º semestre, 0,65, contra 0,46/0,58), e o ganho do grafo é outro: encontrar grupos concretos sem indicar quais variáveis olhar, como a comunidade de maior risco da matrícula (142 alunos, 70% de evasão, 83% com pais de escolaridade desconhecida), um fator que fica no meio do ranking de variáveis isoladas. Com k = 20 e k = 60 o V fica entre 0,32 e 0,36 na matrícula e entre 0,46 e 0,59 no fim do 1º semestre, sempre com p < 0,01, então a conclusão não depende do k = 40.

Números em `resultados/{homofilia,gemeos,variaveis_isoladas,combinacao_fatores,sensibilidade_k,perfis_sucesso}.csv`.
