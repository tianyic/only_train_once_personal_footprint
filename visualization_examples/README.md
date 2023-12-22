# Visulization of dependency graphs for pruning and erasing mode

This visulization of pruning dependency graphs and erasing dependency graphs provides a useful tool for debugging.

In the `pruning` folder, we provide the generated the pruning dependency graphs for the DNNs covered in the [`sanity_check`](https://github.com/tianyic/only_train_once/tree/main/sanity_check)

```python
# view will try to open generated dependency graphs via some pdf reader, set up as False if running on remote servers.
oto.visualize(view=True or False, out_dir=PATH)
```

In the depicted dependency graphs, 

- The nodes marked by the same color forms one node group. The nodes in the same node groups has dependency that needs to pruned together. 

- One node group is **prunable** if it is filled by solid color.

- One node group is **unprunable** if it is outlined by dash lines.

- Nodes with black font color have trainable variables. Otherwise, the font color becomes white.