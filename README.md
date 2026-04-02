# Bird Classifier CUB-200

Classificador de 200 espécies de pássaros usando FastAI e ResNet18.

## Dataset
- CUB-200-2011: 11.788 imagens, 200 espécies
- Dataset: [CUB-200-2011](https://www.kaggle.com/datasets/wenewone/cub2002011)

## Tecnologias
- Python
- FastAI
- ResNet18 (transfer learning)

## Resultados
- Acurácia final: 73.8% (200 espécies)
- 5 épocas de treino

## Como rodar
1. Instale as dependências: `pip install fastai`
2. Baixe o dataset do Kaggle
3. Execute: `python train.py`