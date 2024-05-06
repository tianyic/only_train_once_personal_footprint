# Only Train Once (OTO): Automatic One-Shot DNN Training And Compression Framework

[![OTO-bage](https://img.shields.io/badge/OTO-red?logo=atom&logoColor=white)](#) [![autoML-bage](https://img.shields.io/badge/autoML-blue?logo=dependabot&logoColor=white)](#) [![DNN-training-bage](https://img.shields.io/badge/DNN-training-yellow)](#) [![DNN-compress-bage](https://img.shields.io/badge/DNN-compress-purple)](#) [![Operator-pruning-bage](https://img.shields.io/badge/Operator-pruning-green)](#) [![Operator-erasing-bage](https://img.shields.io/badge/Operator-erasing-CornflowerBlue)](#) [![build-pytorchs-bage](https://img.shields.io/badge/build-pytorch-orange)](#) [![lincese-bage](https://img.shields.io/badge/license-MIT-blue.svg)](#) [![prs-bage](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#)

![oto_overview](https://github.com/tianyic/only_train_once/assets/8930611/131bd6ba-3f94-4b46-8398-074ae311ccf0)

This repository is the (deprecated) Pytorch implementation of **Only-Train-Once** (**OTO**). OTO is an $\color{LimeGreen}{\textbf{automatic}}$, $\color{LightCoral}{\textbf{architecture}}$ $\color{LightCoral}{\textbf{agnostic}}$ DNN $\color{Orange}{\textbf{training}}$ and $\color{Violet}{\textbf{compression}}$ (via $\color{CornflowerBlue}{\textbf{structure pruning}}$ and $\color{DarkGoldenRod}{\textbf{erasing}}$ operators) framework. By OTO, users could train a general DNN either from scratch or a pretrained checkpoint to achieve both high performance and slimmer architecture simultaneously in the one-shot manner (without fine-tuning). 


## Publications

Please find our series of works and [bibtexs](https://github.com/tianyic/only_train_once?tab=readme-ov-file#citation) for kind citations. 

- [OTOv3: Automatic Architecture-Agnostic Neural Network Training and Compression from Structured Pruning to Erasing Operators](https://arxiv.org/abs/2312.09411) preprint.
- [LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery](https://arxiv.org/abs/2310.18356) preprint. 
- [An Adaptive Half-Space Projection Method for Stochastic Optimization Problems with Group Sparse Regularization](https://openreview.net/pdf?id=KBhSyBBeeO) in **TMLR 2023**.  
- [OTOv2: Automatic, Generic, User-Friendly](https://openreview.net/pdf?id=7ynoX1ojPMt) in **ICLR 2023**.
- [Only Train Once (OTO): A One-Shot Neural Network Training And Pruning Framework](https://papers.nips.cc/paper/2021/hash/a376033f78e144f494bfc743c0be3330-Abstract.html) in **NeurIPS 2021**.

![oto_overview_2](https://github.com/tianyic/only_train_once/assets/8930611/ed1f8fda-d43c-4b60-a627-7ce9b2277848) 

In addition, we recommend our following efficient ML works. 

- [DREAM: Diffusion Rectification and Estimation-Adaptive Models](https://www.tianyuding.com/projects/DREAM/), efficient diffusion training, in **CVPR 2024**.
- [DISTILLM: Towards Streamlined Distillation for Large Language Models](https://arxiv.org/pdf/2402.03898.pdf), LLM distillation, in **ICML 2024**.

**Note, we will release the report of HESSO optimizer this June.** Thanks for the interest and support from our community. 

## Installation

We recommend to run the framework under `pytorch>=2.0`. Use `pip` or `git clone` to install.

```bash
pip install only_train_once
```
or
```bash
git clone https://github.com/tianyic/only_train_once.git
```

## Quick Start

We provide an example of OTO framework usage. More explained details can be found in [tutorials](./tutorials/).

### Minimal usage example. 

```python
import torch
from sanity_check.backends import densenet121
from only_train_once import OTO

# Create OTO instance
model = densenet121()
dummy_input = torch.zeros(1, 3, 32, 32)
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

# Create HESSO optimizer
optimizer = oto.hesso(variant='sgd', lr=0.1, target_group_sparsity=0.7)

# Train the DNN as normal via HESSO
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

# A compressed densenet will be generated. 
oto.construct_subnet(out_dir='./')
```

## How the pruning mode in OTO works.

- **Pruning Zero-Invariant Group Partition.** OTO at first automatically figures out the dependancy inside the target DNN to build a pruning dependency graph. Then OTO partitions DNN's trainable variables into so-called Pruning Zero-Invariant Groups (PZIGs). PZIG describes a class of pruning minimally removal structure of DNN, or can be largely interpreted as the minimal group of variables that must be pruned together.
![zig_partition](https://user-images.githubusercontent.com/8930611/224582957-d3955a50-2abc-44b7-b134-1ba0075ca85f.gif)
<!-- ![demonet_dependency_graph_prune](https://github.com/tianyic/only_train_once/assets/8930611/f1100163-4735-4801-a6ed-ccfc326644c6)
![pzigs](https://github.com/tianyic/only_train_once/assets/8930611/b095643c-cb07-4f17-bbbc-c245f2c0cbb4) -->


- **Hybrid Structured Sparse Optimizer.** A structured sparsity optimization problem is formulated. A hybrid structured sparse optimizer, including HESSO, DHSPG, LSHPG, is then employed to find out which PZIGs are redundant, and which PZIGs are important for the model prediction. The selected hybrid optimizer explores group sparsity more reliably and typically achieves higher generalization performance than other sparse optimizers.
![dhspg](https://user-images.githubusercontent.com/8930611/224577550-3814f6c9-0eaf-4d1c-a978-2251b68c2a1a.png)


- **Construct pruned model.** The structures corresponding to redundant PZIGs (being zero) are removed to form the pruned model. Due to the property of PZIGs, **the pruned model returns the exact same output as the full model**. Therefore, **no further fine-tuning** is required. 
<p align="center"><img width="400" alt="comp_construct" src="https://user-images.githubusercontent.com/8930611/224575936-27594b36-1d1d-4daa-9f07-d125dd6e195e.png"></p> 

## Sanity Check

The [`sanity check`](./sanity_check) provides the tests for pruning mode in OTO onto various DNNs from CNN to LLM. The pass of sanity check indicates the compliance of OTO onto target DNN. 
```
python sanity_check/sanity_check.py
```
Note that some tests require additional dependency. Comment off unnecessary tests. We highly recommend to proceed a sanity check over a new customized DNN for testing compliance.  

## Visualization 

The [`visual_examples`](./visual_examples) provides the visualization of pruning dependency graphs and erasing dependency graphs. Visualization serves as a frequently used tool for employing OTO onto new unseen DNNs if meets errors.

## To do list

- Add more explanations into the current repository.

- Release a technical report regarding the [HESSO](https://github.com/tianyic/only_train_once/blob/main/only_train_once/optimizer/hesso.py) optimizer which is not discussed yet in our [papers](https://github.com/tianyic/only_train_once?tab=readme-ov-file#publications).

- Release refactorized DHSPG and LHSPG.

- Release the full pipeline of LoRAShear (upon business administration).

- Provide more tutorials to cover the experiments in the pruning mode. Main experiments in OTOv2 can be found at [otov2_branch](https://github.com/tianyic/only_train_once/tree/otov2_legacy_backup/tutorials).
  
- Release official erasing mode after the review process of OTOv3.

- Provide documentations of the OTO API.


## Welcome Contribution

We would greatly appreciate the contributions in any form, such as bug fixes, new features and new tutorials, from our open-source community. 

We are humble to provide benefits for the AI community. We look forward to working with the community together to make DNN's training and compression to be more automatic and convinient. 

## Open for collabration.

We are open and happy for collabrations. Feel free to reach out <tiachen@microsoft.com> if have any interesting idea.


## Legacy OTOv2 repository

The previous OTOv2 repo has been moved into [legacy_branch](https://github.com/tianyic/only_train_once/tree/otov2_legacy_backup) for academic replication.

## Citation

If you find the repo useful, please kindly star this repository and cite our papers:

```bibtex
For OTOv3 preprint
@article{chen2023otov3,
  title={OTOv3: Automatic Architecture-Agnostic Neural Network Training and Compression from Structured Pruning to Erasing Operators},
  author={Chen, Tianyi and Ding, Tianyu and Zhu, Zhihui and Chen, Zeyu and Wu, HsiangTao and Zharkov, Ilya and Liang, Luming},
  journal={arXiv preprint arXiv:2312.09411},
  year={2023}
}

For LoRAShear preprint
@article{chen2023lorashear,
  title={LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery},
  author={Chen, Tianyi and Ding, Tianyu and Yadav, Badal and Zharkov, Ilya and Liang, Luming},
  journal={arXiv preprint arXiv:2310.18356},
  year={2023}
}

For AdaHSPG+ publication in TMLR (theoretical optimization paper)
@article{dai2023adahspg,
  title={An adaptive half-space projection method for stochastic optimization problems with group sparse regularization},
  author={Dai, Yutong and Chen, Tianyi and Wang, Guanyi and Robinson, Daniel P},
  journal={Transactions on machine learning research},
  year={2023}
}

For OTOv2 publication in ICLR 2023
@inproceedings{chen2023otov2,
  title={OTOv2: Automatic, Generic, User-Friendly},
  author={Chen, Tianyi and Liang, Luming and Tianyu, DING and Zhu, Zhihui and Zharkov, Ilya},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

For OTOv1 publication in NeurIPS 2021
@inproceedings{chen2021otov1,
  title={Only Train Once: A One-Shot Neural Network Training And Pruning Framework},
  author={Chen, Tianyi and Ji, Bo and Tianyu, DING and Fang, Biyi and Wang, Guanyi and Zhu, Zhihui and Liang, Luming and Shi, Yixin and Yi, Sheng and Tu, Xiao},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
