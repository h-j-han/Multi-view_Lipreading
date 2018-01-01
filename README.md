# Multi-view Visual Speech Recognition Based On Multi Task Learning

This repo contains Keras implementation of multi-view visual speech recognition, described in the following paper:

* **H**ou**J**eung **Han**, Sunghun Kang , and Chang D. Yoo, Multi-view Visual Speech Recognition
Based on Multi-task Learning, IEEE International Conference on Image Processing(ICIP), 2017.


![Net11](https://github.com/comc35/Multi-view_Lipreading/blob/master/img/net11.png)


To enhance lipreading, we present a learning method that jointly learns the position of the face and word spoken as well as the words. While mainly training with label of phrases, the model also auxiliarily train with label of facial position.



## Prerequisites
* Linux
* A CUDA-enabled NVIDIA GPU; Recommend video memory >= 11GB


## Getting Started

### Installation
The code requires following dependencies:
* Python 2 
* Theano 0.9.0 ([installation](http://deeplearning.net/software/theano/install_ubuntu.html))
* Keras 1.2.1  ([installation](https://keras.io/#installation))


### Keras backend setup

**Make sure your Keras `backend` is `Theano` and `image_data_format` is `channels_first`**

[How do I check/switch them?](https://keras.io/backend/)


### Download & preprocess dataset

1, Download OuluVS2 dataset [here](http://www.ee.oulu.fi/research/imag/OuluVS2/). (Require registration)
	It's Total 7,800 samples with 52 speakers, ID #1 to #53 except #52.

2, Extract image frames from each ROI video by FFmpeg4 command line tool with option qscale:v=1. mpeg [here](https://mpeg.org/about.html)

3, 20 by 20 pixel color image with maximized the contrast, i.e. all pixel values in each channel are normalized as mapped into [0,1] interval.

4, organize them in `data` folder


### Training & Evaluation

Run `train_test.py` either in your favorite Python IDE or the terminal by typing:

```shell
python train_test.py
```

This would train the model for 200 epochs and save the best model during the training. You can stop it and continue to the evaluation during training if you feel it takes too long. For managing control factor and hyperparamater, you may use - option. For detail, please look `tool.py`.

### Training & Evaluation

* `tools.py` : 
Setting default option or add option.
Function for printing performance

* `plot.py` :
Function for Plot learning curve and confusion matrix and tsne plot

* `model.py` : 
Function for create or load model.
Model architecture can be modified here.

* `data.py` :
Function for processing data such as preprocessing and load data.
Set default directory for data
Learning protocal for training and testing can be set here.


## Note
The implemented is inspired by previous work of D Hyun Lee et al in [here](http://www.ee.oulu.fi/research/imag/OuluVS2/ACCVW.html). The performance might be slightly different from the ones reported in the paper.
