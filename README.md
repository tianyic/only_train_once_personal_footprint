# Only Train Once (OTO): Automatic One-Shot DNN Training And Compression Framework

[![OTO-bage](https://img.shields.io/badge/OTO-red?logo=atom&logoColor=white)](#) [![autoML-bage](https://img.shields.io/badge/autoML-blue?logo=dependabot&logoColor=white)](#) [![DNN-training-bage](https://img.shields.io/badge/DNN-training-yellow)](#) [![DNN-compress-bage](https://img.shields.io/badge/DNN-compress-purple)](#) [![build-pytorchs-bage](https://img.shields.io/badge/build-pytorch-orange)](#) [![build-onnx-bage](https://img.shields.io/badge/build-onnx-green)](#) [![lincese-bage](https://img.shields.io/badge/license-MIT-blue.svg)](#) [![prs-bage](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#)


![otov2_overall0](https://user-images.githubusercontent.com/8930611/224572517-5f284990-e000-4de5-80d2-04900dd672af.png)

This repository is the Pytorch implementation of Only Train Once (OTO). OTO is an automatic general DNN training and compression (via structure pruning) framework. By OTO, users could train a general DNN from scratch to achieve both high performance and slimmer architecture simultaneously in the one-shot manner (without pretraining and fine-tuning). 

## Working Items.

We will release detailed documentations regarding the OTO API in the coming week.

## Publications

Please find our series of works.

- [OTOv2: Automatic, Generic, User-Friendly](https://openreview.net/pdf?id=7ynoX1ojPMt) in ICLR 2023.
- [Only Train Once (OTO): A One-Shot Neural Network Training And Pruning Framework](https://papers.nips.cc/paper/2021/hash/a376033f78e144f494bfc743c0be3330-Abstract.html) in NeurIPS 2021 .

<img width="867" alt="oto_vs_others" src="https://user-images.githubusercontent.com/8930611/224573845-8789c707-db2d-4ba7-9240-f457fddf4359.png">


## Installation

OTO library can be either installed via pip or clone this repository.

```bash
pip install only_train_once
```

or

```bash
git clone https://github.com/tianyic/only_train_once.git
```

## Quick Start

We provide an example of OTO framework usage. More explained details can be found in [tutorals](./tutorials/).

### Minimal usage example. 

```python
import torch
from backends import DemoNet
from only_train_once import OTO

# Create OTO instance
model = DemoNet()
dummy_input = torch.zeros(1, 3, 32, 32)
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

# Create DHSPG optimizer
optimizer = oto.dhspg(lr=0.1, target_group_sparsity=0.7)

# Train the DNN as normal via DHSPG
model.train()
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(max_epoch):
    f_avg_val = 0.0
    for X, y in trainloader:
        X, y = X.cuda(), y.cuda()
        y_pred = model.forward(X)
        f = criterion(y_pred, y)
        optimizer.zero_grad()
        f.backward()
        optimizer.step()

# A DemoNet_compressed.onnx will be generated. 
oto.compress()
```

## How OTO works.

- **Zero-Invariant Group Partition.** OTO at first automatically figures out the dependancy inside the target DNN and partition DNN's trainable variables into so-called Zero-Invariant Groups (ZIGs). ZIG is a class of minimal removal structure of DNN, or can be largely interpreted as the minimal group of variables that must be pruned together. 
![zig_partition](https://user-images.githubusercontent.com/8930611/224582957-d3955a50-2abc-44b7-b134-1ba0075ca85f.gif)


- **Dual Half-Space Project Gradient (DHSPG).** A structured sparsity optimization problem is formulated. DHSPG is then employed to find out which ZIGs are redundant, and which ZIGs are important for the model prediction. DHSPG explores group sparsity more reliably and typically achieves higher generalization performance than other optimizers.
![dhspg](https://user-images.githubusercontent.com/8930611/224577550-3814f6c9-0eaf-4d1c-a978-2251b68c2a1a.png)


- **Construct compressed model.** The structures corresponding to redundant ZIGs (being zero) are removed to form the compressed model. Due to the property of ZIGs, **the compressed model returns the exact same output as the full model**. Therefore, **no further fine-tuning** is required. 
<p align="center"><img width="400" alt="comp_construct" src="https://user-images.githubusercontent.com/8930611/224575936-27594b36-1d1d-4daa-9f07-d125dd6e195e.png"></p> 

## More full and compressed models

Please find more full and compressed models by OTO on [checkpoints](https://drive.google.com/drive/folders/1lZ7Wsehi0hr_g8nztbAFEJIhF8C4Q8Kp?usp=share_link). The full and compressed models return the exact same outputs given the same inputs.

The dependancy graphs for ZIG partition can be found at [Dependancy Graphs](https://drive.google.com/drive/folders/1XVRUEr4cUyT6xVknLF2SsYKgXBZ0gjeD?usp=share_link).

## Remarks and to do list

The current OTO library depends on 

- The target model needed to be convertable into ONNX format for conducting dependancy graph construction.

- Please check our supported [operators](./only_train_once/operation/operators_dict.py) list if meeting some errors.

- The effectiveness (ultimate compression ratio and model performance) relies on the proper usage of DHSPG optimizer. Please go through our [tutorials](./tutorials/) for setup (will be kept updated).

We will routinely complete the following items.

- Provide more tutorials to cover more use cases and applications of OTO. 

- Provide documentations of the OTO API.

- Optimize the dependancy list.

## Welcome Contributions

We greatly appreciate the contributions from our open-source community to make DNN's training and compression to be more automatic and convinient. 

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
