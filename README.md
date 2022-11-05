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

### Evaluation

As a sanity check, run evaluation using our ImageNet **fine-tuned** models:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>
<!-- TABLE BODY -->
<tr><td align="left">fine-tuned checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/1KD5JCj-cdcsPkGPQ9n5hwaSg2Rrvm88i/view?usp=share_link">download</a></td>

</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>1b25e9</tt></td>

</tr>
<tr><td align="left">reference ImageNet accuracy 16x16 patch</td>
<td align="center">83.352</td>

</tr>
</tbody></table>


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
