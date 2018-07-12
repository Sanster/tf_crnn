# CRNN Tensorflow

Original Paper: [An End-to-End Trainable Neural Network for Image-based
Sequence Recognition and Its Application to Scene Text Recognition](http://arxiv.org/abs/1507.05717)

Original Code: http://github.com/bgshih/crnn

## Requirements
```shell
sudo pip3 install -r requirement.txt
```

## Prepare training data
Check [Text Renderer](https://github.com/Sanster/text_renderer) to see how to generate images for training,
or you can download pre-generated image from here [Coming soon]()

## Train
```shell
python3 train.py
```

## Inference
```shell
python3 infer.py
```

## Experiments

Test result on chinese dataset from [caffe-ocr](https://github.com/senlinuc/caffe_ocr)
|Network|Test Acc|Edit Distance|
|-------|--------|-------------|
|PaperCnn|0|0|
|DenseNet|0|0|
|Resnet|0|0|
|SqueezeNet|0|0|

* Edit distance only count error predict results
