# Deep Learning com TensorFlow

Este diretório reúne duas atividades didáticas com TensorFlow/Keras: uma introdução ao
funcionamento de uma rede neural e um experimento de classificação binária com uma MLP.
Os notebooks usam dados pequenos ou sintéticos; não representam um modelo de produção nem
treinam com a base Olist.

## Requisitos

- Python 3.10
- `uv`

Instale as dependências na pasta `deep_learning`:

```powershell
uv sync
```

Abra os notebooks em `notebooks/` com Jupyter:

```powershell
uv run jupyter lab
```

## Atividades

- `notebooks/atividade_tensorflow_rede_neural_atraso_entrega.ipynb` apresenta tensores,
  combinação linear de um neurônio, ReLU, arquitetura de uma MLP, forward pass, probabilidades,
  threshold e binary cross-entropy. É uma atividade de inferência; não chama `model.fit()`.
- `notebooks/rede_neural_tensorflow_backpropagation.ipynb` gera um conjunto sintético,
  separa treino e teste, padroniza os dados sem ajustar o scaler no teste, treina uma MLP com
  Keras e avalia suas previsões.

Os módulos Python em `deep_learning/` ainda são o scaffold do template CCDS e não são usados
pelas atividades dos notebooks.

## Verificação

Os testes executam os dois notebooks em kernels limpos e falham se alguma célula de código
produzir erro:

```powershell
uv run python -m pytest tests
```

O TensorFlow pode exibir avisos de inicialização ou otimização de CPU durante a execução.
