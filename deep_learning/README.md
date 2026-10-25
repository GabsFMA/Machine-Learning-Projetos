# Deep Learning com TensorFlow

Projeto didático para estudar redes neurais artificiais e classificação binária com TensorFlow. O conteúdo atual está concentrado em dois notebooks: um apresenta a propagação direta de uma rede e o outro demonstra treinamento e avaliação com dados sintéticos.

> Este projeto ainda não implementa um pipeline de ponta a ponta com dados reais. Os módulos Python são estruturas iniciais do template Cookiecutter Data Science, e não há datasets nem modelos treinados armazenados na pasta.

## Conteúdo

### `notebooks/atividade_tensorflow_rede_neural_atraso_entrega.ipynb`

Atividade introdutória sobre uma rede neural para estimar atraso de entrega. Utiliza exemplos pequenos com três características (`distancia`, `quantidade_itens` e `frete`) e aborda:

- representação das entradas e do alvo como arrays;
- cálculo de um neurônio por meio de pesos, bias e soma ponderada;
- ativações ReLU e sigmoid;
- arquitetura MLP `3 → 4 → 1` e contagem de parâmetros;
- *forward pass* e interpretação da saída como probabilidade;
- threshold para converter probabilidade em classe;
- Binary Cross-Entropy como função de perda.

O objetivo deste notebook é entender como a rede produz uma previsão e calcula a perda. Os parâmetros não são treinados neste exercício.

### `notebooks/rede_neural_tensorflow_backpropagation.ipynb`

Demonstra um fluxo de treinamento de uma MLP para classificação binária usando dados sintéticos gerados com `make_classification`. O notebook divide os dados em treino e teste, padroniza as features, treina o modelo com `model.fit()`, acompanha a loss de treino e validação e apresenta métricas e matriz de confusão.

O treinamento e o cálculo dos gradientes são realizados pelo TensorFlow; o notebook não implementa o algoritmo de backpropagation manualmente. Os dados utilizados também não são dados reais de entregas da Olist.

## Estrutura

```text
deep_learning/
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── deep_learning/
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── plots.py
│   └── modeling/
│       ├── predict.py
│       └── train.py
├── docs/                    # Esqueleto de documentação MkDocs
├── models/                  # Ainda sem modelos treinados
├── notebooks/               # Atividades didáticas
├── reports/
├── tests/
├── Makefile
├── pyproject.toml
└── uv.lock
```

Os módulos em `deep_learning/` ainda contêm código de exemplo do template. Eles não implementam o pré-processamento, a engenharia de características, o treinamento ou a inferência descritos nos notebooks.

## Ambiente

O projeto declara Python `~=3.10.0` e utiliza `uv` para gerenciar as dependências. Para sincronizar o ambiente:

```powershell
uv sync
```

Para iniciar o JupyterLab:

```powershell
uv run jupyter lab
```

Abra um dos notebooks em `notebooks/` e execute suas células no ambiente do projeto. A versão do TensorFlow registrada no `uv.lock` é 2.21.0; a saída salva no notebook introdutório indica uma execução anterior com TensorFlow 2.20.0.
