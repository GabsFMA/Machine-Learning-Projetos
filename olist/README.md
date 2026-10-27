# Olist — previsão de atraso de entregas

Projeto de Ciência de Dados com o conjunto público de e-commerce da Olist. O objetivo é estimar, **no momento da aprovação do pagamento**, o risco de um pedido ser entregue depois da data prometida.

O notebook de análise exploratória registra 96.456 pedidos entregues com as datas necessárias para definir o resultado. Nesse recorte, 7.826 pedidos (8,11%) foram entregues com atraso e 88.630 (91,89%) dentro do prazo ou antes. Como as classes são desbalanceadas, acurácia isolada não é suficiente para avaliar os modelos.

## Dados

Os arquivos originais ficam em `data/raw/`:

| Arquivo | Registros | Conteúdo |
| --- | ---: | --- |
| `olist_orders_dataset.csv` | 99.441 | Pedidos, status e datas |
| `olist_order_items_dataset.csv` | 112.650 | Itens, vendedores, preços e frete |
| `olist_customers_dataset.csv` | 99.441 | Identificação e localização dos clientes |
| `olist_geolocation_dataset.csv` | 1.000.163 | Coordenadas e localização por CEP |
| `olist_order_payments_dataset.csv` | 103.886 | Forma, parcelas e valor dos pagamentos |
| `olist_order_reviews_dataset.csv` | 99.224 | Notas e comentários das avaliações |
| `olist_products_dataset.csv` | 32.951 | Categorias e atributos dos produtos |
| `olist_sellers_dataset.csv` | 3.095 | Identificação e localização dos vendedores |
| `product_category_name_translation.csv` | 71 | Tradução dos nomes de categorias |

O pipeline atual utiliza pedidos, itens e clientes. As demais tabelas estão disponíveis para análises e extensões futuras.

O arquivo `data/interim/dataset.csv` contém a base consolidada usada nas análises: 96.456 linhas, uma por pedido, e 15 colunas. A pasta `data/interim/` também recebe a saída do pipeline descrito abaixo.

## Preparação dos dados

O módulo `module_olist/main.py` carrega pedidos, itens e clientes e:

1. mantém pedidos com status `delivered` e datas de aprovação, entrega real e entrega estimada preenchidas;
2. cria o alvo `is_late`, que vale `1` quando a entrega real ocorre depois da data estimada, e `0` caso contrário;
3. agrega os itens por pedido, calculando quantidade de itens e vendedores, preço total e frete total;
4. junta essas informações aos dados do cliente, preservando uma linha por pedido;
5. cria as features `promised_days`, `purchase_month`, `purchase_weekday` e `purchase_hour`.

As datas de envio e entrega real são usadas para construir o histórico e o alvo; não fazem parte das features utilizadas na previsão. A análise exploratória também verifica essa separação para reduzir o risco de *data leakage*.

Por padrão, o resultado é salvo em `data/interim/orders_dataset_refined.csv`.

## Modelagem

Os pipelines em `module_olist/modeling/` comparam Gradient Boosting, XGBoost e LightGBM. O pré-processamento mantém as features numéricas e aplica *one-hot encoding* às categóricas. A validação cruzada estratificada usa cinco folds e compara métricas como precisão, *recall*, F1, ROC-AUC e PR-AUC; o threshold de classificação é escolhido para maximizar o F1 das previsões *out-of-fold*.

O artefato atualmente armazenado em `models/` é um pipeline LightGBM (`best_model.joblib`) com threshold `0.14`, registrado em `models/metadata.json`. O módulo `module_olist/modeling/inference.py` demonstra inferência em uma amostra de cinco pedidos.

## Configuração e execução

O projeto requer Python **3.12.1** e usa `uv` para gerenciar o ambiente e as dependências:

```powershell
uv sync
```

Para gerar a base refinada a partir dos CSVs originais:

```powershell
uv run python -m module_olist.main
```

Para executar a demonstração de inferência, primeiro gere `data/interim/orders_dataset_refined.csv`:

```powershell
uv run python -m module_olist.modeling.inference
```

## Notebooks e relatórios

- [`notebooks/02_EDA_template.ipynb`](notebooks/02_EDA_template.ipynb): análise do alvo, construção da base por pedido e exploração de prazos, complexidade do pedido, localização e sazonalidade.
- [`notebooks/orders_data_quality.ipynb`](notebooks/orders_data_quality.ipynb): validação da tabela de pedidos com um contrato Pandera para completude, unicidade, validade e consistência temporal.
- [`notebooks/orders_eda.ipynb`](notebooks/orders_eda.ipynb): geração do relatório de perfil dos pedidos.
- `reports/reports.html`: relatório HTML de perfil da tabela de pedidos.

## Estrutura do projeto

```text
olist/
├── data/
│   ├── raw/                 # CSVs originais
│   └── interim/             # Bases intermediárias e refinadas
├── docs/                    # Documentação MkDocs
├── models/                  # Modelo treinado e metadados
├── notebooks/               # EDA e qualidade dos dados
├── reports/                 # Relatório HTML e figuras
├── module_olist/
│   ├── config.py            # Caminhos e configuração
│   ├── dataset.py           # Leitura, target, agregação e junções
│   ├── features.py          # Engenharia de features
│   ├── main.py              # Pipeline de preparação dos dados
│   └── modeling/            # Pipelines, validação e inferência
├── tests/                   # Testes automatizados
├── Makefile                 # Comandos auxiliares
├── pyproject.toml           # Dependências e configuração
└── uv.lock                 # Versões resolvidas das dependências
```

## Testes

Para executar os testes:

```powershell
uv run python -m pytest tests
```

Os testes cobrem a criação do alvo, a agregação dos itens, o salvamento do dataset e o cálculo das métricas de avaliação dos modelos.
