# Visualization of dependency graphs for pruning and erasing mode

This visualization of pruning dependency graphs and erasing dependency graphs provides a frequently used tool for employing OTO onto new unseen DNNs if meets errors.

In the [`pruning`](https://github.com/tianyic/only_train_once/tree/main/visual_examples/pruning) folder, we provide the generated pruning dependency graphs for the DNNs covered in the [`sanity_check`](https://github.com/tianyic/only_train_once/tree/main/sanity_check) along with some dependency graphs met during our daily use of OTO onto various applications. 

```python
# view will try to open generated dependency graphs via some pdf reader, set up as False if running on remote servers.
oto.visualize(view=True or False, out_dir=PATH)
```

In the depicted pruning dependency graphs, 

- The nodes marked by the same color form one node group. The nodes in the same node group have dependency that need to be pruned together. 

- One node group is **prunable** if it is filled by solid color.

- One node group is **unprunable** if it is outlined by dash lines.

- Nodes with black font color have trainable variables. Otherwise, the font color becomes white.


We will provide more explanations for the visualization [`erasing`](https://github.com/tianyic/only_train_once/tree/main/visual_examples/erasing) mode. 