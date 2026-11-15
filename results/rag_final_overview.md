# Visão Final Consolidada dos Resultados — Olist VoC RAG

> Gerado em: `2026-11-15T16:24:58.155051+00:00`

## 1. Resumo Executivo do Corpus e Ingestão

- **Total de avaliações únicas (`olist_enriched.parquet`)**: 98.410
- **Avaliações com texto substantivo indexadas (`chunks.parquet` / ChromaDB / BM25)**: 41.405
- **Avaliações apenas com nota**: 57.005
- **Nota média global**: 4.09 / 5,0
- **Taxa global de atraso na entrega**: 7.76%
- **Janela temporal**: `2016-09` a `2018-10` (74 categorias, 8 status de pedido)
- **Auditoria de limpeza**: `99224` brutas -> `814` duplicadas removidas -> `56040` sem texto + `965` ruídos excluídos do índice vetorial -> `1` PII(s) LGPD mascarado(s).

## 2. Impacto Quantitativo de Atrasos, Categorias e Status

- **Entregas no prazo (`90777` avaliações)**: nota média **4.22**
- **Entregas atrasadas (`7633` avaliações)**: nota média **2.57**

### Top 5 Categorias em Volume

| Categoria | Avaliações | Nota Média | Taxa de Atraso |
| --- | ---: | ---: | ---: |
| `cama_mesa_banho` | 9172 | 3.98 | 8.57% |
| `beleza_saude` | 8712 | 4.19 | 8.69% |
| `esporte_lazer` | 7605 | 4.17 | 7.52% |
| `informatica_acessorios` | 6598 | 4.03 | 7.40% |
| `moveis_decoracao` | 6278 | 4.03 | 8.24% |

### Categorias com Menor Nota Média (>= 100 avaliações)

| Categoria | Avaliações | Nota Média | Taxa de Atraso |
| --- | ---: | ---: | ---: |
| `nao_informada` | 2107 | 3.17 | 5.46% |
| `moveis_escritorio` | 1250 | 3.63 | 8.96% |
| `fashion_roupa_masculina` | 110 | 3.70 | 2.73% |
| `audio` | 343 | 3.84 | 12.83% |
| `construcao_ferramentas_seguranca` | 161 | 3.89 | 4.35% |

## 3. Resultados da Ablação de Recuperação Híbrida

| Estratégia | Casos | Recall@5 | Recall@10 | MRR | Latência Média (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 Only | 8 | 0.5000 | 0.4375 | 0.5012 | 57.5 |
| Dense Vector Only | 8 | 0.3250 | 0.3500 | 0.5188 | 253.8 |
| Hybrid (Dense + BM25 + RRF) | 8 | 0.3750 | 0.4250 | 0.6031 | 106.8 |
| Full Pipeline (Multi-Query + Metadata Filter + RRF + MMR) | 8 | 0.9000 | 0.8500 | 1.0000 | 318.3 |

## 4. Principais Achados Qualitativos Recuperados pelo RAG

### Atrasos e Logística (Nota <= 2) (max_dense=0.7605)

Resumo Quantitativo (Parquet Olist — recorte ativo):
- Total de avaliações no recorte: 4,126 (2,872 com comentário textual)
- Nota média no recorte: 1.14 / 5.0
- Taxa de entrega atrasada no recorte: 100.0%
- Top 3 categorias em volume: cama_mesa_banho (442), beleza_saude (394), esporte_lazer (321)

- `[89a0e3830a71ad7ec14efa88211092f5]` (nota=1, `brinquedos`, status=`delivered`, 2017-12): "Já se passou 4 dias do prazo máximo de entrega e até agora nada. No total já são mais de 30 dias aguardando meu produto. O mesmo foi enviado via correio ao invés de transportadora. Insatisfeita."
- `[350e4b450fe67e009abfdcfd6e172e13]` (nota=1, `automotivo`, status=`delivered`, 2017-10): "Demora pra envio e ainda não chegou, problema pra despachar e tbm pra envio da parte dos correios"
- `[aeae0c849feace30fae1d1abfa3517c3]` (nota=1, `cama_mesa_banho`, status=`delivered`, 2017-11): "É a primeira vez que um entrega do baratheon não é entregue dentro do prazo.Não sei se foi a transportadora que atrasou e também não importa, comprei do baratheon e a empresa que é responsável."

### Qualidade e Avarias em Móveis e Decoração (Nota <= 2) (max_dense=0.6922)

Resumo Quantitativo (Parquet Olist — recorte ativo):
- Total de avaliações no recorte: 1,020 (733 com comentário textual)
- Nota média no recorte: 1.23 / 5.0
- Taxa de entrega atrasada no recorte: 27.5%
- Top 3 categorias em volume: moveis_decoracao (1020)

- `[5b2602b06d6360ecf96806983c564cb8]` (nota=1, `moveis_decoracao`, status=`delivered`, 2017-08): "O produto veio com defeito, a cortina veio com a embalagem violada e o produto estava danificado, queimado. Fiz a reclamação direto com a loja e eles ficaram de me retornar mas não retornaram até hoj"
- `[6abbf7cd7559eb5b32d6ef5cb25db317]` (nota=1, `moveis_decoracao`, status=`delivered`, 2018-03): "Chegou com a caixa machucada . Ao abrir , o produto avariado . Tentaram maquiar o vidro quebrado que estava por baixo , mas eu vi . Pior compra que já fiz , vergonhoso ."
- `[17c8b61567d5f3a10f0c2312bd72845d]` (nota=1, `moveis_decoracao`, status=`delivered`, 2017-05): "A arara é de qualidade ruim, fragil o que fez com que o mesmo ja chegasse quebrada"

### Pedidos Cancelados (max_dense=0.7957)

Resumo Quantitativo (Parquet Olist — recorte ativo):
- Total de avaliações no recorte: 577 (392 com comentário textual)
- Nota média no recorte: 1.75 / 5.0
- Taxa de entrega atrasada no recorte: 0.2%
- Top 3 categorias em volume: nao_informada (149), esporte_lazer (46), beleza_saude (35)

- `[f5866b458dc0cd28b894fd7ac9a650b8]` (nota=1, `bebes`, status=`canceled`, 2017-07): "O pedido foi cancelado 5 dias antes da entrega por indisponibilidade no estoque. Estou aguardando meu reembolso."
- `[c8953fdac2721068fda086e9af911db0]` (nota=1, `relogios_presentes`, status=`canceled`, 2017-12): "Minha compra foi cancelada por falto do producto no stok e ainda não devolveron minho dinheiro. Para que colocam esso na venta cuando não esta en existencia no stok"
- `[ca13795db8e305829cb406ef608d87f9]` (nota=1, `utilidades_domesticas`, status=`canceled`, 2017-06): "Recebi um comunicado que não teria o produto no estoque da loja, foi feito o cancelamento e estou aguardando o estorno do valor pago"

### Elogios e Entregas no Prazo (Nota 5) (max_dense=0.7509)

Resumo Quantitativo (Parquet Olist — recorte ativo):
- Total de avaliações no recorte: 55,213 (20,479 com comentário textual)
- Nota média no recorte: 5.00 / 5.0
- Taxa de entrega atrasada no recorte: 0.0%
- Top 3 categorias em volume: beleza_saude (5182), cama_mesa_banho (4740), esporte_lazer (4493)

- `[26d72a4208c2a9b675e8d5307429000d]` (nota=5, `cama_mesa_banho`, status=`delivered`, 2018-01): "O produto chegou bem antes do prazo e é de excelente qualidade."
- `[fdd6de525e1abb9a69d588475673d867]` (nota=5, `telefonia`, status=`delivered`, 2018-08): "Super recomendado — O produto chegou antes do prazo e de excelente qualidade."
- `[9f9b3681b361132d16cacedf636ecd4c]` (nota=5, `beleza_saude`, status=`delivered`, 2017-12): "Produto foi entregue bem antes do prazo. Chegou bem condicionado, excelente qualidade de atendimento."
