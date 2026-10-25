# Classificador de imagens CIFAR-10

Projeto de classificação de imagens com uma rede neural convolucional (CNN) implementada em TensorFlow/Keras. O modelo reconhece as dez categorias da base CIFAR-10: avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio e caminhão.

## Requisitos

- Python 3.10
- `uv`
- Acesso à internet na primeira execução para baixar a CIFAR-10, caso ela ainda não esteja no cache do Keras

Instale e sincronize as dependências na pasta `image_classifier`:

```powershell
uv sync
```

## Uso

Baixe/carregue a base e confira as dimensões:

```powershell
uv run python -m image_classifier.dataset
```

Treine a CNN, avalie-a no conjunto de teste e salve o modelo:

```powershell
uv run python -m image_classifier.main train
```

O comando usa por padrão 10 épocas, batch size 64 e 20% dos dados de treino para validação. O modelo é salvo em `models/cifar10_cnn.keras`. Os parâmetros podem ser ajustados:

```powershell
uv run python -m image_classifier.main train --epochs 20 --batch-size 128 --validation-split 0.15
```

Classifique uma imagem do conjunto de teste usando o modelo salvo:

```powershell
uv run python -m image_classifier.main predict-sample --index 0
```

Para treinar ou carregar um modelo em outro caminho, use `--model-path` nos comandos `train` e `predict-sample`.

## Arquitetura e fluxo

As imagens RGB têm dimensão `32 × 32 × 3` e são normalizadas para o intervalo `[0, 1]`. Os rótulos inteiros são mantidos no formato esperado por `sparse_categorical_crossentropy`.

A CNN contém três blocos convolucionais com 16, 32 e 64 filtros, cada um seguido de *max pooling*. As features resultantes passam por uma camada densa de 64 neurônios e uma saída `softmax` de dez classes. O treinamento usa o otimizador Adam e acompanha acurácia e loss.

O fluxo executável está em `image_classifier/main.py`:

1. carrega a CIFAR-10 pelo Keras;
2. treina o modelo com uma parcela de validação;
3. avalia loss e acurácia no conjunto de teste;
4. salva o modelo no formato Keras `.keras`;
5. permite classificar uma amostra do conjunto de teste com o modelo salvo.

## Testes e lint

Execute os testes:

```powershell
uv run python -m pytest tests
```

Verifique o estilo:

```powershell
uv run ruff check
uv run ruff format --check
```

## Estrutura

```text
image_classifier/
├── data/                    # Diretórios de dados (CIFAR-10 é gerenciada pelo cache Keras)
├── docs/                    # Documentação MkDocs
├── image_classifier/
│   ├── dataset.py            # Carregamento e normalização da CIFAR-10
│   ├── main.py               # Comandos de treino e inferência
│   ├── plots.py              # Visualização de imagens e histórico
│   └── modeling/
│       ├── model.py          # Arquitetura e compilação da CNN
│       ├── train.py          # Treinamento
│       └── predict.py        # Avaliação, persistência e predições
├── models/                   # Modelo treinado (gerado pelo comando train)
├── tests/                    # Testes automatizados
├── Makefile
├── pyproject.toml
└── uv.lock
```
