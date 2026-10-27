# Geração de legendas com BLIP

Este projeto contém um notebook interativo que usa o modelo pré-treinado
`Salesforce/blip-image-captioning-base` para gerar legendas de imagens. O experimento
compara a geração livre com a geração condicionada por prefixo e permite testar
prefixos adicionais na mesma foto.

O notebook não treina o modelo, não gera imagens e não usa o Qwen-Image-Edit mencionado
na seção final; essa aplicação externa é apenas apresentada como exemplo separado.

## Execução

Abra [`multimodal_generation_revisado.ipynb`](./multimodal_generation_revisado.ipynb) no
Google Colab ou em um ambiente Jupyter com suporte a `%pip`. Execute as células em ordem.
O notebook instala as dependências, baixa os pesos do BLIP na primeira execução e salva
os resultados na pasta `outputs_blip/`, além de gerar um arquivo ZIP.

O notebook usa uma imagem pública por padrão. Para fornecer uma imagem própria, altere
`USAR_UPLOAD` para `True`; essa opção de upload requer Google Colab. GPU é opcional, mas
pode reduzir o tempo de inferência. A primeira carga baixa aproximadamente 990 MB de pesos.

## Avaliação e limitações

A avaliação automática inclui cobertura de palavras-chave na legenda, e há uma rubrica
para conferência humana. As notas humanas estão inicialmente vazias e precisam ser
preenchidas na célula correspondente. A cobertura lexical é apenas uma medida simples:
não substitui a avaliação de correção, completude ou detalhes inventados.

As saídas já incorporadas ao notebook são resultados de uma execução anterior; execute-o
novamente para reproduzir os resultados no seu ambiente. A instalação fixa versões de
algumas dependências e pode conflitar com pacotes preexistentes em ambientes Colab usados
por outros projetos; prefira uma sessão limpa.
