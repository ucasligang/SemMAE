# Introduction

Paper accepted at NeurIPS 2022.

This is a official repository of SemMAE.
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

This implementation is in PyTorch+GPU. 
* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.
It maybe needed for the repositories: tensorboard. It can be installed by 'pip install '.

### Pretrained models

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">800-epochs</th>
<th valign="bottom">ViT-Base 16x16 patch</th>
<th valign="bottom">ViT-Base 8x8 patch</th>
<!-- TABLE BODY -->
<tr><td align="left">fine-tuned checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/1GaGWNv8I-ADF8e-Bvftgr2k8qNeyLdTJ/view?usp=share_link">download</a></td>
  <td align="center"><a href="">waiting</a></td>

</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>1482ae</tt></td>
<td align="center"><tt>waiting</tt></td>
</tr>

</tbody></table>

### Evaluation

As a sanity check, run evaluation using our ImageNet **fine-tuned** models:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">800-epochs</th>
<th valign="bottom">ViT-Base 16x16 patch</th>
<th valign="bottom">ViT-Base 8x8 patch</th>
<!-- TABLE BODY -->
<tr><td align="left">fine-tuned checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/1KD5JCj-cdcsPkGPQ9n5hwaSg2Rrvm88i/view?usp=share_link">download</a></td>
  <td align="center"><a href="">waiting</a></td>

</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>bbc5ef</tt></td>
<td align="center"><tt>waiting</tt></td>
</tr>
<tr><td align="left">reference ImageNet accuracy</td>
<td align="center">83.352</td>
<td align="center">waiting</td>
</tr>
</tbody></table>


Evaluate ViT-Base in a single GPU (`${IMAGENET_DIR}` is a directory containing `{train, val}` sets of ImageNet):
```
python main_finetune.py --eval --resume SemMAE_epoch799_vit_base_checkpoint-99.pth --model vit_base_patch16 --batch_size 16 --data_path ${IMAGENET_DIR}
```
This should give:
```
* Acc@1 83.352 Acc@5 96.494 loss 0.745
Accuracy of the network on the 50000 test images: 83.4%
```
Note that all of our results were obtained on the 800epoches setting

## Contact

This repo is currently maintained by Gang Li([@ucasligang](https://github.com/ucasligang)).
