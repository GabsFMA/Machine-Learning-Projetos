# Atividade 01 — Cookiecutter Data Science (CCDS)

Esta atividade teve como objetivo conhecer e explorar o **Cookiecutter Data Science (CCDS)**, uma ferramenta utilizada para criar estruturas padronizadas para projetos de Ciência de Dados e Machine Learning.

Durante a atividade, foi criado um projeto utilizando o CCDS para observar, na prática, como a ferramenta organiza automaticamente diretórios, arquivos de configuração, código-fonte, dados, notebooks, modelos e documentação.

## Objetivo da atividade

O principal objetivo foi compreender a importância de uma estrutura organizada e reproduzível no desenvolvimento de projetos de Machine Learning.

A atividade permitiu observar como o CCDS separa diferentes responsabilidades do projeto, como:

- armazenamento dos dados;
- processamento dos dados;
- desenvolvimento de notebooks;
- treinamento e armazenamento de modelos;
- geração de relatórios;
- criação de código-fonte reutilizável;
- gerenciamento de dependências;
- testes e documentação.

## Criação do projeto

O projeto foi inicializado utilizando o CCDS e recebeu o nome:

```text
olist
```

Como exemplo de contexto para o projeto, foi utilizada a descrição:

```text
Brazilian ecommerce public dataset
```

A partir das opções selecionadas durante a criação, o CCDS gerou automaticamente a estrutura inicial do projeto.

## Gerenciamento do ambiente

O projeto utiliza um ambiente virtual próprio:

```text
.venv
```

As dependências são gerenciadas utilizando o `uv`.

Para sincronizar o ambiente do projeto:

```bash
uv sync
```

Para ativar manualmente o ambiente virtual:

```bash
source .venv/bin/activate
```

Durante a atividade também foi adicionada a biblioteca `seaborn`:

```bash
uv add seaborn
```

## Estrutura gerada pelo CCDS

A organização gerada segue aproximadamente a seguinte estrutura:

```text
olist/
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
│
├── docs/
├── models/
├── notebooks/
├── references/
├── reports/
│
├── module_olist/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── train.py
│   └── plots.py
│
├── tests/
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

## Principais diretórios

### `data/`

Responsável pelo armazenamento dos conjuntos de dados utilizados no projeto.

O CCDS divide os dados de acordo com o estágio de processamento:

- `raw/`: dados originais, sem modificações;
- `external/`: dados provenientes de fontes externas;
- `interim/`: dados intermediários durante transformações;
- `processed/`: dados já preparados para utilização nos modelos.

Essa separação ajuda a manter os dados originais preservados e torna o pipeline de transformação mais fácil de acompanhar.

### `notebooks/`

Destinado aos notebooks utilizados durante exploração de dados, testes, experimentos e desenvolvimento inicial das soluções.

### `models/`

Utilizado para armazenar modelos treinados, previsões ou outros artefatos relacionados aos modelos de Machine Learning.

### `reports/`

Destinado aos resultados produzidos durante as análises, como relatórios, gráficos e figuras.

### `references/`

Armazena materiais auxiliares utilizados no projeto, como documentação, dicionários de dados e referências externas.

### `module_olist/`

Contém o código-fonte principal do projeto.

A separação do código em módulos permite evitar que toda a lógica fique concentrada em notebooks e facilita sua reutilização.

Alguns dos arquivos gerados incluem:

- `dataset.py`: código relacionado à obtenção e preparação dos dados;
- `features.py`: criação e transformação de características;
- `plots.py`: funções relacionadas à geração de visualizações;
- `modeling/train.py`: código relacionado ao treinamento dos modelos;
- `modeling/predict.py`: código relacionado à utilização dos modelos treinados.

### `tests/`

Destinado aos testes automatizados do projeto.

### `pyproject.toml`

Arquivo responsável pelas configurações do projeto e pelo gerenciamento de suas dependências.

### `Makefile`

Permite definir comandos que automatizam tarefas recorrentes do projeto.

## Por que utilizar uma estrutura como essa?

Projetos de Machine Learning podem crescer rapidamente e envolver diferentes etapas, como:

```text
dados
  ↓
tratamento
  ↓
análise exploratória
  ↓
criação de features
  ↓
treinamento
  ↓
avaliação
  ↓
modelo
  ↓
relatórios
```

Sem uma organização definida, notebooks, datasets, modelos e scripts podem acabar misturados.

O CCDS busca resolver esse problema oferecendo uma estrutura inicial padronizada, facilitando:

- organização;
- manutenção;
- colaboração;
- reprodutibilidade;
- separação de responsabilidades;
- evolução do projeto.

## Conclusão

A atividade permitiu conhecer o funcionamento do **Cookiecutter Data Science** e compreender como uma estrutura padronizada pode auxiliar na organização de projetos de Machine Learning.

Mais do que apenas criar pastas automaticamente, o CCDS propõe uma forma de separar as diferentes etapas de um projeto de Ciência de Dados, tornando seu desenvolvimento mais organizado e reproduzível.
