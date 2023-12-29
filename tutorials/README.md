# Tutorials

We will routinely update tutorials to cover varying use cases in 2024. Please expect slow update in Januaray due to recent heavy workload. 

Here are the **empirical principles** that we would like to highlight if employing OTO onto new applications outside our tutorials.

## Sanity Check

We highly recommend to proceed a sanity check to test the compliance of OTO onto target DNN. The sanity check will randomly pick up a set of minimally removal structures as redundant 

```python
oto.random_set_zero_groups()
```
and produce compact subnetwork, as presented in [sanity_check](https://github.com/tianyic/only_train_once/blob/main/sanity_check/test_resnet18.py). If sanity check does not pass, please mark illed node groups as unprunable via

```python
oto.mark_unprunable_by_node_ids()
```
e.g., in [YOLOv5](https://github.com/tianyic/only_train_once/blob/main/sanity_check/test_yolov5.py).


## Optimizer setup (Important)

OTO is designed to **seamless integrate into the existing training pipeline for the full model**, which is typically reliable to achieve high performance given full model.

We empirically recommend to set up the [**hyperparameters**](https://github.com/tianyic/only_train_once/blob/cbb3d3dccf95c383e9cddcbaf8592cf3db13817b/only_train_once/__init__.py#L47) in OTO's optimizers **exactly the same as the baseline optimizers** for performance. Please note varying optimizer setup may result in significantly different performance. Meanwhile, some applications require a longer training steps for convergence due to the weaker learning capacity of sparse model. 

## Old tutorials 

Tutorials over old library can be found at [here](https://github.com/tianyic/only_train_once/tree/otov2_legacy_backup/tutorials). It covers ResNet50 CIFAR10, ResNet50 ImageNet and VGG16BN CIFAR10. These tutorials will be refreshed upon the new library next year. 