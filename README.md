# scene_classifier
This repository is scene classifier on Jetson Nano.

## dataset.py
PyTorchで学習時にデータを供給するためのクラス．

## preprocess.py
モデル学習時に前処理を行うクラス．

## train.py
モデルを学習させるコード．

----------
モデルの最適化は実行したいGPU上で行う必要があるため，
以下はJetson Nano上でのみ動作させる必要があります．

## torch2trt
学習済みのモデルをJetsonNanoに最適化したモデルに変換するコード．

## recognition_trt.py
最適化したモデルを用いてUSBカメラからの画像を実際に推論するコード．現状では商店街or交差点の2値分類になっています．