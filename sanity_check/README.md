# Sanity Check

We highly recommend to proceed a sanity check to test the compliance of OTO onto target DNN. The sanity check will randomly pick up a set of minimally removal structures as redundant 

```python
oto.random_set_zero_groups()
```
and produce compact subnetwork, as presented in [sanity_check](https://github.com/tianyic/only_train_once/blob/main/sanity_check/test_resnet18.py). If sanity check does not pass, please mark illed node groups as unprunable via

```python
oto.mark_unprunable_by_node_ids()
```
For example, in [YOLOv5](https://github.com/tianyic/only_train_once/blob/main/sanity_check/test_yolov5.py), we mark the node groups corresponding to detection heads as unprunable.

If all variable groups of pruning minimally removal structures are pruning zero-invariant groups (PZIGs), the returned sub-network should return the exact same output as the full group sparse model given random inputs.

Run the sanity check by the below command

```python
python sanity_check.py
```

Note some sanity checks may require additional dependency, thereby comment off the ones that you do not need. 