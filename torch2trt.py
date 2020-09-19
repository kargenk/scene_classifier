#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import tensorrt as trt

from torchvision import models

# 警告やエラーなどをとらえるロガー
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def torch2onnx(save_onnx_path, net):
    """ torchのモデルをONNX形式に変換． """

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    input_names = ['input_1']
    output_names = ['output_1']

    torch.onnx.export(
        net, dummy_input, save_onnx_path, verbose=True,
        input_names=input_names, output_names=output_names,
        export_params=True
    )


def build_trt_engine(onnx_model_path, engine_path):
    """ ONNXモデルをbuildしてplan fileに変換． """

    # TensorRT 7からは必要
    network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # TensorRTエンジンとONNXモデルパーサを初期化
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(network_creation_flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 各種設定
    builder.max_workspace_size = 1 << 30  # 2GB，TensorRTエンジンで使用するGPUメモリの最大値
    builder.max_batch_size = 1            # バッチサイズ
    # builder.fp16_mode = True              # fp16を用いる場合(デフォルトはINT8)，ビルド時間は伸びる

    # ONNXモデルのパース
    with open(onnx_model_path, 'rb') as model:
        print('Beginning ONNX model parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX model')

    # 何かエラーがあれば止める
    if parser.num_errors > 0:
        print(parser.get_error(0).desc())
        raise Exception

    # TensorRTエンジンの生成
    print('Building an engine ...')
    engine = builder.build_cuda_engine(network)
    # context = engine.create_execution_context()
    print('Completed creating Engine')

    # TensorRTエンジンの保存
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
        print('Completed saving TRT engine')


if __name__ == '__main__':

    # モデルをインスタンス化し，推論モードに変更後，学習済み重みをロード
    net = models.resnet18(pretrained=True)
    net.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 2),
        nn.Softmax(dim=1),  # 出力を確率にするために追加
    )
    net.load_state_dict(torch.load('./models/epoch_50.pth'))
    net.eval()

    # モデルをGPUに載せる
    device = torch.device('cuda')
    net = net.to(device)

    # 保存先
    onnx_path = './models/scene_classifier.onnx'
    engine_path = './models/scene_classifier.engine'

    # torch -> ONNX -> TensorRT
    torch2onnx(onnx_path, net)
    build_trt_engine(onnx_path, engine_path)
