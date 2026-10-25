# Primeiros passos

Instale Python 3.10 e `uv`, depois sincronize as dependências na pasta do projeto:

```powershell
uv sync
```

Organize imagens consentidas em `data/authorized/<identidade>/` e imagens para consulta diretamente em `data/test/`. Confira os arquivos encontrados:

```powershell
uv run python -m face_recognition.dataset
```

Compare uma imagem com a galeria:

```powershell
uv run python -m face_recognition.main recognize --image data/test/consulta.jpg
```

Processe o diretório de teste e salve um relatório CSV:

```powershell
uv run python -m face_recognition.main recognize-test-set
```

Na primeira execução, o DeepFace pode baixar os pesos do modelo escolhido. Consulte o README para opções, estrutura dos dados e limitações de uso.
