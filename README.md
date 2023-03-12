# Only Train Once (OTO): A One-Shot Neural Network Training And Pruning Framework

This repository is the Pytorch implementation of Only Train Once (OTO). OTO is an automatic general DNN training and compression (via structure pruning) framework. 

## Publications

Please find our series of works.

- [OTOv2: Automatic, Generic, User-Friendly](https://openreview.net/pdf?id=7ynoX1ojPMt) in ICLR 2023.
- [Only Train Once (OTO): A One-Shot Neural Network Training And Pruning Framework](https://papers.nips.cc/paper/2021/hash/a376033f78e144f494bfc743c0be3330-Abstract.html) in NeurIPS 2021 .


<img width="1105" alt="overview" src="https://user-images.githubusercontent.com/8930611/144922447-843b6a40-4fa3-4af7-85d0-62cc43d1b4ca.png">

## Installation

```bash
pip install only_train_once
```

or

```bash
git clone https://github.com/tianyic/only_train_once.git
```

<!-- ## Quick Start

We provide an example of OTO framework usage. More explained details can be found in [tutorals](./tutorials/).

### 0. How it works
 -->

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
<!-- 
## Zero-Invariant Group (ZIG)

Zero-invariant groups serve as one of two fundamental components to OTO. A ZIG has an attractive property is that if equaling to zero, then the corresponding structure contributes null to the model output, thereby can be directly removed. ZIG is generic to various DNN architectures, such as Conv-Layer, Residual Block, Fully-Connected Layer and Multi-Head Attention Layer as follows.

<img width="995" alt="zig_conv_bn" src="https://user-images.githubusercontent.com/8930611/144923778-3a31718f-5f0e-42cc-a0a9-357aae463700.png">
<img width="959" alt="zig_residual" src="https://user-images.githubusercontent.com/8930611/144923631-b1f7a4f5-6bd5-4003-be44-2275b9cfa69d.png">
<img width="836" alt="zig_fc_multi_head" src="https://user-images.githubusercontent.com/8930611/144923967-3458d322-8998-469d-874b-1d59475c0490.png">


## Half-Space Projected Gradient Descent Method (HSPG)

Half-Space Projected Gradient Descent Method serve as another fundamental component to OTO to promote more ZIGs as zero. Hence, redundant structures can be pruned without retraining. HSPG utilizes a novel Half-Space projection operator to yield group sparsity, which is more effective than the standard proximal method because of a larger projection region. 

<img width="1025" alt="hspg" src="https://user-images.githubusercontent.com/8930611/144924639-1e0b6f36-92bf-4f09-80a8-9e5b3fb9b1d4.png"> -->
