#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from preprocess import ImageTransform
from dataset import SceneDataset, make_datapath_list


def get_model4tl():
    """転移学習用のモデルを返す．"""

    net = models.resnet18(pretrained=True)

    # 学習させるパラメータ(最終層)以外は勾配計算をなくし，変化しないように設定
    for param in net.parameters():
        param.requires_grad = False

    # シーン分類用に最終層を変更
    net.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 2),
    )

    net.train()
    print('学習済み重みをロードし，訓練モードに設定しました')

    # # 転移学習で学習させるパラメータ
    # params_to_update = []

    # # 学習させるパラメータ以外は勾配計算をなくし，変化しないように設定
    # for name, param in net.named_parameters():
    #     if name in target_params:
    #         param.requires_grad = True
    #         params_to_update.append(param)
    #         print(name)
    #     else:
    #         param.requires_grad = False

    # print('-' * 20)
    # print(params_to_update)

    return net


def train(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    print('device:', device)

    # ネットワークがある程度固定であれば，高速化させる
    torch.backends.cudnn.benchmark = True

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            # 未学習時の性能検証用にepoch:0の訓練はしない
            if (epoch == 1) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 各イテレーションの値を加算
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            # 各epochでのlossと正解率を表示
            num_data = len(dataloaders_dict[phase].dataset)
            epoch_loss = epoch_loss / num_data
            epoch_acc = epoch_corrects.double() / num_data

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        if (epoch % 50 == 0) and (epoch != 0):
            save_path = './models/epoch_{}.pth'.format(epoch)
            torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    index = 0
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 32

    # ファイルパス取得
    train_list = make_datapath_list(phase='train')
    val_list = make_datapath_list(phase='val')
    print('train:', len(train_list), 'val:', len(val_list))

    # 前処理クラスとデータセット，データローダの定義
    image_transform = ImageTransform(size, mean, std)
    train_dataset = SceneDataset(train_list, image_transform, 'train')
    val_dataset = SceneDataset(val_list, image_transform, 'val')
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    # 転移学習のネットワーク
    net = get_model4tl()

    # 損失関数と最適化手法
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4)

    # 訓練
    train(net, dataloaders_dict, criterion, optimizer, num_epochs=2)
