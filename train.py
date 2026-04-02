import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastai.vision.all import *

def main():
    # Caminho do dataset
    path = Path('C:/datasets/cub200/CUB_200_2011/images')

    # Criar o DataBlock
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms()
    ).dataloaders(path, bs=32, num_workers=0)

    # Treinar o modelo
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(5)

    # Salvar o modelo
    learn.save('cub200_model')
    print("Modelo salvo!")

if __name__ == '__main__':
    main()