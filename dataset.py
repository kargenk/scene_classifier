#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from preprocess import ImageTransform


class SceneDataset(Dataset):
    """
    シーン画像(buildings，street)のデータセットクラス．

    Attributes
    ----------
    file_list : list
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'val'
        前処理のモード管理フラグ
    """

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        """画像の枚数を返す"""
        return len(self.file_list)

    def __getitem__(self, index):
        """前処理をした画像のTensor形式のデータとラベルを取得"""

        # index番目の画像をロードして前処理を行う
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)

        # 画像ラベルをフォルダ名から抜き出して数値化
        label = img_path.split('/')[2]
        if label == 'buildings':
            label = 0
        elif label == 'street':
            label = 1

        return img_transformed, label


def make_datapath_list(phase='train', data_dir='./data/*/'):
    """
    データへのパスを格納したリストを作成する関数．

    Parameters
    ----------
    phase : 'train' or 'val'
        訓練データか検証データかの指定フラグ

    Returns
    ----------
    path_list : list
        データへのパスを格納したリスト
    """

    target_path = os.path.join(data_dir + phase + '/*.jpg')
    print(target_path)

    # サブディレクトリまで探査
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path.replace('\\', '/'))  # for Windows

    return path_list


if __name__ == '__main__':
    # ファイルパス取得
    train_list = make_datapath_list(phase='train')
    val_list = make_datapath_list(phase='val')
    print('train:', len(train_list), 'val:', len(val_list))

    # Datasetのテスト
    index = 0
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    image_transform = ImageTransform(size, mean, std)
    train_dataset = SceneDataset(train_list, image_transform, 'train')
    val_dataset = SceneDataset(val_list, image_transform, 'val')
    print(train_dataset.__getitem__(index)[0].shape)  # Size([3, 224, 224])
    print(train_dataset.__getitem__(index)[1])        # 0: buildings

    # Dataloaderのテスト
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    batch_iterator = iter(dataloaders_dict['train'])
    inputs, labels = next(batch_iterator)
    print(inputs.shape)  # Size([b=32, 3, 224, 224])
    print(labels)
