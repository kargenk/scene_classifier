#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


class ImageTransform():
    """
    画像の前処理クラス．訓練時と検証時では異なる動作をする(訓練時のみデータ拡張)．
    画像のサイズをリサイズし，色を標準化する．

    Attributes
    ----------
    resize : int
        リサイズ後の大きさ
    mean : (R, G, B)
        各色チャネルの平均値
    std : (R, G, B)
        各色チャネルの標準偏差
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            'val': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモード管理フラグ
        """
        return self.data_transform[phase](img)


if __name__ == '__main__':
    # 画像の読み込み
    image_file_path = './data/buildings/train/4.jpg'
    img = Image.open(image_file_path)

    # 元画像の表示
    plt.imshow(img)
    plt.show()

    # 画像の前処理と処理後画像の表示
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = ImageTransform(size, mean, std)
    img_transformed = transform(img, phase='train')

    # (RGB, H, W) -> (H, W, RGB)に変換して，値を[0, 1]にクリッピングして表示
    img_transformed = img_transformed.numpy().transpose((1, 2, 0))
    img_transformed = np.clip(img_transformed, 0, 1)
    plt.imshow(img_transformed)
    plt.show()
