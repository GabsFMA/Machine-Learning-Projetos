# Primeiros passos

## Preparar o ambiente

Instale Python 3.10 e `uv`. Na pasta `image_classifier`, sincronize as dependências:

```powershell
uv sync
```

## Treinar e salvar o modelo

```powershell
uv run python -m image_classifier.main train
```

Na primeira execução, o Keras pode baixar a CIFAR-10. O modelo treinado será salvo em `models/cifar10_cnn.keras`.

## Fazer uma predição

Depois do treinamento, classifique uma imagem do conjunto de teste:

```powershell
uv run python -m image_classifier.main predict-sample --index 0
```

O índice pode ser alterado para selecionar outra imagem do conjunto de teste. Use `--model-path` para indicar um modelo salvo em outro local.

## Validar

```powershell
uv run python -m pytest tests
uv run ruff check
uv run ruff format --check
```
