# Reconhecimento facial local

Projeto demonstrativo de identificação facial em um conjunto fechado de pessoas cadastradas, usando a biblioteca DeepFace. As imagens de consulta são comparadas com imagens organizadas localmente por identidade; o projeto não treina uma rede própria nem persiste embeddings faciais.


## Requisitos e instalação

- Python 3.10
- `uv`
- Acesso à internet na primeira execução para baixar os pesos do modelo DeepFace, se ainda não estiverem no cache

Na pasta `face_recognition`, instale as dependências:

```powershell
uv sync
```

O projeto declara DeepFace, TensorFlow/Keras e OpenCV como dependências. O modelo de representação padrão é `Facenet`, com limiar e métrica fornecidos pelo DeepFace para esse modelo e a distância cosseno.

## Preparar as imagens

Organize as imagens conhecidas em subpastas, uma por identidade:

```text
data/
├── authorized/
│   ├── pessoa_a/
│   │   ├── imagem_1.jpg
│   │   └── imagem_2.png
│   └── pessoa_b/
│       └── imagem_1.jpg
└── test/
    ├── consulta_1.jpg
    └── consulta_2.png
```

São aceitos arquivos `.jpg`, `.jpeg`, `.png` e `.bmp`. Cada imagem deve conter um rosto visível; o detector OpenCV e a verificação de face estão habilitados. A galeria e as imagens de teste ficam em `data/`, diretório ignorado pelo Git para evitar versionar imagens pessoais.

Confira quantas imagens foram encontradas:

```powershell
uv run python -m face_recognition.dataset
```

## Reconhecer uma imagem

Compare uma imagem com a galeria:

```powershell
uv run python -m face_recognition.main recognize --image data/test/consulta_1.jpg
```

O resultado mostra a identidade correspondente ou informa que a pessoa não foi reconhecida. É possível selecionar outro modelo DeepFace e outra pasta de galeria:

```powershell
uv run python -m face_recognition.main recognize --image data/test/consulta_1.jpg --model-name Facenet512 --authorized-dir data/authorized
```

## Processar o conjunto de teste

Para comparar todas as imagens diretamente dentro de `data/test/` com a galeria:

```powershell
uv run python -m face_recognition.main recognize-test-set
```

O relatório padrão é `reports/recognition_results.csv`, com o caminho da imagem, identidade reconhecida (em branco quando não há correspondência), distância, limiar e indicador de reconhecimento. Esse relatório também é ignorado pelo Git por conter resultados associados a imagens pessoais. Para alterar diretórios e destino:

```powershell
uv run python -m face_recognition.main recognize-test-set --test-dir data/test --authorized-dir data/authorized --output-path reports/resultados.csv
```

Esse comando **não calcula acurácia**: o diretório de teste não contém rótulos esperados; ele apenas registra as correspondências produzidas pelo DeepFace. Comparações são feitas com cada imagem cadastrada, então o custo e a possibilidade de falsos positivos crescem com o tamanho da galeria.

## Testes e lint

Os testes usam imagens temporárias vazias e substituem a chamada ao DeepFace por um dublê; não requerem fotografias nem download dos pesos do modelo.

```powershell
uv run python -m pytest tests
uv run ruff check
uv run ruff format --check
```

## Estrutura

```text
face_recognition/
├── data/                    # Imagens locais (não versionadas)
│   ├── authorized/           # Galeria: uma subpasta por identidade
│   └── test/                 # Imagens de consulta
├── docs/                    # Documentação MkDocs
├── face_recognition/
│   ├── config.py             # Diretórios e configuração padrão
│   ├── dataset.py            # Descoberta das imagens locais
│   ├── main.py               # Comandos de reconhecimento e relatório
│   └── modeling/
│       └── predict.py        # Comparação de pares via DeepFace
├── reports/                  # Relatórios gerados (resultados ignorados pelo Git)
├── tests/
├── Makefile
├── pyproject.toml
└── uv.lock
```
