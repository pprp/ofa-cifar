# Once for All - CIFAR10

[TOC]

## Introduction

[Once for all](https://github.com/mit-han-lab/once-for-all) is an one-stage one-shot Neural Architecture Search Algorithm, which mainly support ImageNet Datasets. 

In this repository, most codes are from https://github.com/mit-han-lab/once-for-all.

We mainly focus on training OFA(Once for all) on CIFAR10 dataset. 

What we do:

- Support CIFAR10 dataloader
- Modify training codes
- Support Single GPU Training
- Rewrite code about Max Teachernet Training
- Release TeacherNet weight(Coming soon..)



## How to train **OFA Networks**

```bash
mpirun -np 32 -H <server1_ip>:8,<server2_ip>:8,<server3_ip>:8,<server4_ip>:8 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python train_ofa_net.py
```

or 

```bash
horovodrun -np 32 -H <server1_ip>:8,<server2_ip>:8,<server3_ip>:8,<server4_ip>:8 \
    python train_ofa_net.py
```

## Requirement

* Python 3.6+
* Pytorch 1.4.0+
* ImageNet Dataset 
* Horovod

## How to use / evaluate **OFA Networks**
### Use
```python
""" OFA Networks.
    Example: ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
""" 
from ofa.model_zoo import ofa_net
ofa_network = ofa_net(net_id, pretrained=True)
    
# Randomly sample sub-networks from OFA network
ofa_network.sample_active_subnet()
random_subnet = ofa_network.get_active_subnet(preserve_weight=True)
    
# Manually set the sub-network
ofa_network.set_active_subnet(ks=7, e=6, d=4)
manual_subnet = ofa_network.get_active_subnet(preserve_weight=True)
```
If the above scripts failed to download, you download it manually from [Google Drive](https://drive.google.com/drive/folders/10leLmIiMtaRu4J46KwrBaMydvQt0qFuI?usp=sharing) and put them under $HOME/.torch/ofa_nets/.

### Evaluate

```
python eval_ofa_net.py --path 'Your path to imagenet' --net ofa_mbv3_d234_e346_k357_w1.0 
```

## How to use / evaluate **OFA Specialized Networks** 
### Use
```python
""" OFA Specialized Networks.
Example: net, image_size = ofa_specialized('flops@595M_top1@80.0_finetune@75', pretrained=True)
""" 
from ofa.model_zoo import ofa_specialized
net, image_size = ofa_specialized(net_id, pretrained=True)
```
If the above scripts failed to download, you download it manually from [Google Drive](https://drive.google.com/drive/folders/1ez-t_DAHDet2fqe9TZUTJmvrU-AwofAt?usp=sharing) and put them under $HOME/.torch/ofa_specialized/.

### Evaluate
```
python eval_specialized_net.py --path 'Your path to imagent' --net flops@595M_top1@80.0_finetune@75 
```

![](figures/cnn_imagenet_new.png)



```BibTex
@inproceedings{
  cai2020once,
  title={Once for All: Train One Network and Specialize it for Efficient Deployment},
  author={Han Cai and Chuang Gan and Tianzhe Wang and Zhekai Zhang and Song Han},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://arxiv.org/pdf/1908.09791.pdf}
}
```






