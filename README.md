# Machine Learning

Repositório central destinado ao armazenamento e organização dos trabalhos, desafios e atividades desenvolvidos durante a disciplina de **Machine Learning**.

A proposta deste repositório é manter todos os projetos da disciplina em um único local, utilizando subpastas independentes para cada trabalho realizado ao longo do semestre.

## Organização do repositório

Cada atividade possui sua própria pasta, contendo os arquivos, códigos, dados, notebooks, modelos e documentação necessários para seu desenvolvimento.

Cada subprojeto possuirá um `README.md` próprio com informações específicas, como:

- objetivo da atividade;
- descrição da base de dados;
- técnicas de Machine Learning utilizadas;
- decisões tomadas durante o desenvolvimento;
- métricas e resultados;
- instruções para execução.

## Trabalhos

| Projeto | Descrição |
|---|---|
| [Olist](./olist/) | Previsão de atraso na entrega de pedidos usando dados do e-commerce Olist. Inclui preparação dos dados, comparação de modelos, avaliação e explicações SHAP. |
| [Deep Learning](./deep_learning/) | Atividades didáticas com TensorFlow/Keras: conceitos de redes neurais e treinamento de uma MLP para classificação binária em dados sintéticos. |
| [Classificador de imagens](./image_classifier/) | Rede neural convolucional com TensorFlow/Keras para classificar imagens da base CIFAR-10 em dez categorias. |
| [Reconhecimento facial](./face_recognition/) | Identificação de pessoas em uma galeria local de imagens usando DeepFace, com suporte a reconhecimento de imagens individuais ou de um conjunto de teste. |
| [Geração de legendas com BLIP](./multimodal_generation_blip/) | Notebook que usa o modelo pré-treinado BLIP para gerar legendas de imagens e comparar geração livre e condicionada por prefixos. |

Cada projeto tem sua própria pasta e README com instruções de execução, dependências e detalhes específicos.

## Ambientes e dependências

Cada trabalho poderá possuir seu próprio ambiente virtual e arquivo de dependências.

Em projetos utilizando `uv`, por exemplo, o ambiente pode ser reproduzido executando:

```bash
uv sync
```

Os ambientes virtuais locais, como `.venv`, não são versionados no repositório.
