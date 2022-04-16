# Conformer-- A Combination of Vision Transformer and Convolution Neural Network

This repository is for NYU ECE-7123 deep learning Mini-Project1. Team members:

- Name: Shunyi Zhu  NetID:sz3719
- Name:Zhen Wang   NetID:zw2655
- Name: Haoze He     NetID: hh2537



### Abstract 

Both convolution neural network(CNN) and vision transformer(ViT) are success in past few years.
They are two dominant frameworks in the field of computer vision. However, these two frameworks
have their own drawbacks. As for CNN, the convolution layer has limited respective field, which
makes it hard to capture the associated information between global and local area. As for ViT, it can
capture the associated information using self-attention. But without similarity comparisons among
all tokens also leads to high redundancy. In addition, the low-efficiency flaw of ViT transformer
caused by high computational/space complexity in Self-Attention. To combine the advantage of
both ViT and CNN, we plan to reduce the number of tokens using submodular optimization, merge
tokens with high similarity into one, and use it as input for CNN architecture.



The code will be implemented using Pytorch. The code baseline is the pytorch implementation of paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.



This paper show that Transformers applied directly to image patches and pre-trained on large datasets work really well on image recognition task.



![fig1](./img/figure1.png)

Vision Transformer achieve State-of-the-Art in image recognition task with standard Transformer encoder and fixed-size patches. In order to perform classification, author use the standard approach of adding an extra learnable "classification token" to the sequence.

![fig2](./img/figure2.png)



## Warning

The code is built based on ViT and will make improvement based on it.



## Usage

### 1. Download Pre-trained model (Google's Official Checkpoint)
* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), R50+ViT-B_16(**97.96M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**), ViT-H_14(**630.8M**)
  * imagenet21k pre-train models
    * ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14
  * imagenet21k pre-train + imagenet2012 fine-tuned models
    * ViT-B_16-224, ViT-B_16, ViT-B_32, ViT-L_16-224, ViT-L_16, ViT-L_32
  * Hybrid Model([Resnet50](https://github.com/google-research/big_transfer) + Transformer)
    * R50-ViT-B_16
```
# imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz

# imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz

```

### 2. Train Model
```
python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
```
CIFAR-10 and CIFAR-100 are automatically download and train. In order to use a different dataset you need to customize [data_utils.py](./utils/data_utils.py).

The default batch size is 512. When GPU memory is insufficient, you can proceed with training by adjusting the value of `--gradient_accumulation_steps`.

Also can use [Automatic Mixed Precision(Amp)](https://nvidia.github.io/apex/amp.html) to reduce memory usage and train faster
```
python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --fp16 --fp16_opt_level O2
```



## Results
To verify that the converted model weight is correct, we simply compare it with the author's experimental results. We trained using mixed precision, and `--fp16_opt_level` was set to O2.

### imagenet-21k
* [**tensorboard**](https://tensorboard.dev/experiment/Oz9GmmQIQCOEr4xbdr8O3Q)

|    model     |  dataset  | resolution | acc(official) | acc(this repo) |  time   |
| :----------: | :-------: | :--------: | :-----------: | :------------: | :-----: |
|   ViT-B_16   | CIFAR-10  |  224x224   |       -       |     0.9908     | 3h 13m  |
|   ViT-B_16   | CIFAR-10  |  384x384   |    0.9903     |     0.9906     | 12h 25m |
|   ViT_B_16   | CIFAR-100 |  224x224   |       -       |     0.923      |  3h 9m  |
|   ViT_B_16   | CIFAR-100 |  384x384   |    0.9264     |     0.9228     | 12h 31m |
| R50-ViT-B_16 | CIFAR-10  |  224x224   |       -       |     0.9892     | 4h 23m  |
| R50-ViT-B_16 | CIFAR-10  |  384x384   |     0.99      |     0.9904     | 15h 40m |
| R50-ViT-B_16 | CIFAR-100 |  224x224   |       -       |     0.9231     | 4h 18m  |
| R50-ViT-B_16 | CIFAR-100 |  384x384   |    0.9231     |     0.9197     | 15h 53m |
|   ViT_L_32   | CIFAR-10  |  224x224   |       -       |     0.9903     | 2h 11m  |
|   ViT_L_32   | CIFAR-100 |  224x224   |       -       |     0.9276     |  2h 9m  |
|   ViT_H_14   | CIFAR-100 |  224x224   |       -       |      WIP       |         |


### imagenet-21k + imagenet2012
* [**tensorboard**](https://tensorboard.dev/experiment/CXOzjFRqTM6aLCk0jNXgAw/#scalars)

|    model     |  dataset  | resolution |  acc   |
| :----------: | :-------: | :--------: | :----: |
| ViT-B_16-224 | CIFAR-10  |  224x224   |  0.99  |
| ViT_B_16-224 | CIFAR-100 |  224x224   | 0.9245 |
|   ViT-L_32   | CIFAR-10  |  224x224   | 0.9903 |
|   ViT-L_32   | CIFAR-100 |  224x224   | 0.9285 |


### shorter train
* In the experiment below, we used a resolution size (224x224).
* [**tensorboard**](https://tensorboard.dev/experiment/lpknnMpHRT2qpVrSZi10Ag/#scalars)

|  upstream   |  model   |  dataset  | total_steps /warmup_steps | acc(official) | acc(this repo) |
| :---------: | :------: | :-------: | :-----------------------: | :-----------: | :------------: |
| imagenet21k | ViT-B_16 | CIFAR-10  |          500/100          |    0.9859     |     0.9859     |
| imagenet21k | ViT-B_16 | CIFAR-10  |         1000/100          |    0.9886     |     0.9878     |
| imagenet21k | ViT-B_16 | CIFAR-100 |          500/100          |    0.8917     |     0.9072     |
| imagenet21k | ViT-B_16 | CIFAR-100 |         1000/100          |    0.9115     |     0.9216     |


## Visualization
The ViT consists of a Standard Transformer Encoder, and the encoder consists of Self-Attention and MLP module.
The attention map for the input image can be visualized through the attention score of self-attention.

Visualization code can be found at [visualize_attention_map](./visualize_attention_map.ipynb).

![fig3](./img/figure3.png)


## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [Pytorch Image Models(timm)](https://github.com/rwightman/pytorch-image-models)


## Citations

```bibtex
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
