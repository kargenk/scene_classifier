#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# tensorrtの推論用の共通関数
sys.path.append('/usr/src/tensorrt/samples/python/')
import common

from preprocess import ImageTransform

# 前処理用の定数
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = ImageTransform(size, mean, std)

# モデルと学習済み重みの読み込み
model_path = './models/epoch_50.pth'
net = models.resnet18(pretrained=False)
net.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 2),
        nn.Softmax(dim=1),  # 出力を確率にするために追加
    )
load_weights = torch.load(model_path, map_location=torch.device('cpu'))
net.load_state_dict(load_weights)
net.eval()  # 推論モード


def load_engine(engine_path):
    """ trtのエンジンを読み込む """

    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    return engine, context


def inference(engine, context, imgs_transformeds):
    """ ResNet-18で推論 """

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    inputs[0].host = imgs_transformeds.numpy()  # bites-like, not Tensor
    trt_outputs = common.do_inference(context, bindings=bindings,
                                      inputs=inputs, outputs=outputs,
                                      stream=stream)
    return torch.Tensor(trt_outputs)


def main():
    # モデルの読み込み
    engine, context = load_engine('./models/scene_classifier.engine')

    # カメラからの画像を推論
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        sys.exit()

    # FPS計測開始
    fps = 0
    count = 0
    max_count = 10
    tm = cv2.TickMeter()
    tm.start()

    while cap.isOpened():
        ret, frame = cap.read()

        # resize the window
        window_size = (256, 256)
        frame = cv2.resize(frame, window_size)

        # 推論結果の出力
        frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_pil)
        frame_transformed = transform(frame_pil, phase='val')
        output = inference(engine, context, frame_transformed.unsqueeze(0))
        print(output)

        pred_class = ''
        _, pred_id = torch.max(output, dim=1)
        if pred_id == 0:
            pred_class = 'intersection'
            print('pred: 交差点')
        elif pred_id == 1:
            pred_class = 'shopping street'
            print('pred: 商店街')

        if count == max_count:
            tm.stop()
            fps = max_count / tm.getTimeSec()
            tm.reset()
            tm.start()
            count = 0

        cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv2.putText(frame, 'pred: {}'.format(pred_class), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), thickness=2)
        cv2.imshow('frame', frame)
        count += 1

        # キーの受付で終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
