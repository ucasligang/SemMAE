# Introduction

Paper accepted at NeurIPS 2022.

This is a official repository of SemMAE.
We will open our models and code soon.
Our code references the [MAE](https://github.com/facebookresearch/mae), thanks a lot for their outstanding work!
For details of our work see [Semantic-Guided Masking for Learning Masked Autoencoders](https://arxiv.org/pdf/2206.10207.pdf). 

<div align="center">
  <img width="900", src="https://github.com/ucasligang/SemMAE/blob/main/src/figure1.png">
</div>

## Citation

```
@article{li2022semmae,
  title={SemMAE: Semantic-Guided Masking for Learning Masked Autoencoders},
  author={Li, Gang and Zheng, Heliang and Liu, Daqing and Wang, Chaoyue and Su, Bing and Zheng, Changwen},
  journal={arXiv preprint arXiv:2206.10207},
  year={2022}
}
```

Evaluate ViT-Base in a single GPU (`${IMAGENET_DIR}` is a directory containing `{train, val}` sets of ImageNet):
```
python main_finetune.py --eval --resume mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path ${IMAGENET_DIR}
```
This should give:
```
* Acc@1 83.352 Acc@5 96.494 loss 0.745
Accuracy of the network on the 50000 test images: 83.4%
```
Note that all of our results were obtained on the 800epoches setting

## Contact

This repo is currently maintained by Gang Li([@ucasligang](https://github.com/ucasligang)).
