# Classificador de imagens CIFAR-10

Projeto de visão computacional organizado com base no formato Cookiecutter Data Science
(CCDS). Implementa uma rede neural convolucional com TensorFlow/Keras para classificar imagens
da base CIFAR-10 em dez categorias: avião, automóvel, pássaro, gato, cervo, cachorro, sapo,
cavalo, navio e caminhão.

O fluxo é implementado como pacote Python e comandos CLI; não depende de um notebook para
treinamento ou inferência. O Keras baixa a CIFAR-10 na primeira execução e mantém os dados em
seu cache local, sem versionar as imagens no repositório.

## Requisitos

- Python 3.10
- `uv`
- Acesso à internet na primeira execução para baixar a CIFAR-10

Na pasta `image_classifier`, instale as dependências:

```powershell
uv sync
```

## Fluxo de uso

Confira o carregamento da base e as dimensões dos conjuntos:

```powershell
uv run python -m image_classifier.dataset
```

Treine a CNN, avalie-a no conjunto de teste e salve o modelo:

```powershell
uv run python -m image_classifier.main train
```

O comando usa 10 épocas, batch size 64 e 20% dos dados de treino para validação por padrão.
É possível ajustar esses valores e o caminho do modelo:

```powershell
uv run python -m image_classifier.main train --epochs 20 --batch-size 128 --validation-split 0.15
```

O modelo é salvo em `models/cifar10_cnn.keras`. Para classificar uma imagem do conjunto de
teste com o artefato salvo:

```powershell
uv run python -m image_classifier.main predict-sample --index 0
```

Use `--model-path` nos comandos `train` e `predict-sample` para indicar outro caminho de
persistência. O índice de `predict-sample` seleciona a imagem do conjunto de teste e deve
estar dentro dos limites desse conjunto.

## Dados e modelo

As imagens CIFAR-10 têm formato `32 × 32 × 3` e são normalizadas para o intervalo `[0, 1]`.
Os rótulos são achatados para inteiros, conforme esperado pela função
`sparse_categorical_crossentropy`.

A arquitetura em `image_classifier/modeling/model.py` contém três blocos convolucionais com
16, 32 e 64 filtros, cada um seguido por *max pooling*. As features passam por uma camada
densa de 64 neurônios e uma saída `softmax` de dez classes. O treino usa Adam e registra
acurácia e loss; o fluxo também avalia no conjunto de teste e persiste o modelo no formato
Keras `.keras`.

## Testes, lint e documentação

Execute os testes automatizados:

```powershell
uv run python -m pytest tests
```

Verifique lint e formatação:

```powershell
uv run ruff check
uv run ruff format --check
```

A documentação complementar fica em `docs/docs/`. Para servi-la localmente ou gerar a versão
estática, execute a partir de `image_classifier/docs`:

```powershell
uv run mkdocs serve
uv run mkdocs build
```

## Organização CCDS

```text
image_classifier/
├── docs/                         # Documentação MkDocs
│   └── docs/
├── image_classifier/             # Pacote-fonte
│   ├── config.py                 # Caminhos do projeto
│   ├── dataset.py                # Carregamento e normalização CIFAR-10
│   ├── features.py               # Scaffold CCDS não utilizado no fluxo CIFAR-10
│   ├── main.py                   # CLI de treino e inferência
│   ├── plots.py                  # Visualizações de dados e treinamento
│   └── modeling/
│       ├── model.py              # Arquitetura e compilação da CNN
│       ├── train.py              # Treinamento
│       └── predict.py            # Avaliação, persistência e predições
├── models/                       # Artefatos de modelo gerados localmente
├── notebooks/                    # Reservado para notebooks; não necessário ao fluxo
├── references/                   # Materiais de referência
├── reports/figures/              # Figuras geradas
├── tests/                        # Testes automatizados
├── Makefile                      # Atalhos para dados, treino, teste e inferência
├── pyproject.toml                # Dependências e configuração do pacote
└── uv.lock                       # Dependências resolvidas
```
