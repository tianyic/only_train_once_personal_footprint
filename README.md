# Only Train Once (OTO): Automatic One-Shot DNN Training And Compression Framework

![otov2_overall0](https://user-images.githubusercontent.com/8930611/224572517-5f284990-e000-4de5-80d2-04900dd672af.png)

This repository is the Pytorch implementation of Only Train Once (OTO). OTO is an automatic general DNN training and compression (via structure pruning) framework. 

## Publications

Please find our series of works.

- [OTOv2: Automatic, Generic, User-Friendly](https://openreview.net/pdf?id=7ynoX1ojPMt) in ICLR 2023.
- [Only Train Once (OTO): A One-Shot Neural Network Training And Pruning Framework](https://papers.nips.cc/paper/2021/hash/a376033f78e144f494bfc743c0be3330-Abstract.html) in NeurIPS 2021 .

<img width="867" alt="oto_vs_others" src="https://user-images.githubusercontent.com/8930611/224573845-8789c707-db2d-4ba7-9240-f457fddf4359.png">


## Installation

OTO library can be either install via pip or clone this repository.

```bash
pip install only_train_once
```

or

```bash
git clone https://github.com/tianyic/only_train_once.git
```

## Quick Start

We provide an example of OTO framework usage. More explained details can be found in [tutorals](./tutorials/).

### 0. How OTO works

- **Zero-Invariant Group Partition.** OTO at first automatically figures out the dependancy inside the target DNN and partition DNN's trainable variables into so-called Zero-Invariant Groups (ZIGs). ZIG is a class of minimal removal structure of DNN, or can be largely interpreted as the minimal group of variables that must be pruned together. 
![Presentation1](https://user-images.githubusercontent.com/8930611/224577449-fb2a1ae6-3e29-4a3f-9136-9deb73746741.gif)


- **Dual Half-Space Project Gradient (DHSPG).** A structured sparsity optimization problem is formulated. DHSPG is then employed to find out each ZIGs are redundant, which ZIGs are important for the model prediction. DHSPG explores group sparsity more reliably and typically achieves higher genelizarion performance than other optimizers.

![dhspg_git](https://user-images.githubusercontent.com/8930611/224575607-3a6f204e-23c6-4cc2-86b7-075d82e33080.png)


- **Construct compressed model.** The structures corresponding to redundant ZIGs (being zero) are removed to form the compressed model. Due to the property of ZIGs, **the compressed model return the exact same output as the full model**, thereby **no further fine-tuning** being required. 
<p align="center"><img width="400" alt="comp_construct" src="https://user-images.githubusercontent.com/8930611/224575936-27594b36-1d1d-4daa-9f07-d125dd6e195e.png"></p> 


## Citation

If you find the repo useful, please kindly star this repository and cite our papers:

```
@inproceedings{chen2023otov2,
  title={OTOv2: Automatic, Generic, User-Friendly},
  author={Chen, Tianyi and Liang, Luming and Tianyu, DING and Zhu, Zhihui and Zharkov, Ilya},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{chen2021only,
  title={Only Train Once: A One-Shot Neural Network Training And Pruning Framework},
  author={Chen, Tianyi and Ji, Bo and Tianyu, DING and Fang, Biyi and Wang, Guanyi and Zhu, Zhihui and Liang, Luming and Shi, Yixin and Yi, Sheng and Tu, Xiao},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```


## Half-Space Projected Gradient Descent Method (HSPG)

Half-Space Projected Gradient Descent Method serve as another fundamental component to OTO to promote more ZIGs as zero. Hence, redundant structures can be pruned without retraining. HSPG utilizes a novel Half-Space projection operator to yield group sparsity, which is more effective than the standard proximal method because of a larger projection region. 

<img width="1025" alt="hspg" src="https://user-images.githubusercontent.com/8930611/144924639-1e0b6f36-92bf-4f09-80a8-9e5b3fb9b1d4.png"> 
