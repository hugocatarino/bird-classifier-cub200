import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastai.vision.all import *

if __name__ == '__main__':
    path = Path('C:/datasets/cub200/CUB_200_2011/images')

    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms()
    ).dataloaders(path, bs=32, num_workers=0)

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.load('cub200_model')
    img = PILImage.create('C:/Users/HugoC/Downloads/teste.jpg')
    pred, idx, probs = learn.predict(img)

    print(f"Espécie previsa: {pred}")
    print(f"Confiança: {probs[idx]:.2%}")