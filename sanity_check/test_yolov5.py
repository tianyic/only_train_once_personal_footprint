import torch
from only_train_once import OTO
import unittest
import os
import onnxruntime as ort
import numpy as np

OUT_DIR = './cache'

class TestYolov5(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 640, 640)):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # All parameters in the pretrained Yolov5 are not trainable.
        for _, param in model.named_parameters():
            param.requires_grad = True

        oto = OTO(model, dummy_input)
        # Mark a conv-concat and the detection heads as unprunable 
        # The node_ids may be varying upon different torch version. 
        oto.mark_unprunable_by_node_ids(
            # ['node-229', 'node-329', 'node-443', 'node-553'] 
            ['node-229', 'node-581', 'node-471', 'node-359']
        )
        # The above can be also achieved by 
        oto.mark_unprunable_by_param_names(
            ['model.model.model.9.cv1.conv.weight', 'model.model.model.24.m.2.weight', \
             'model.model.model.24.m.1.weight', 'model.model.model.24.m.0.weight']
        )
        
        # Display param name and shape in dependency graph visualization
        oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)
        # Compute FLOP and param for full model. 
        full_flops = oto.compute_flops(in_million=True)['total']
        full_num_params = oto.compute_num_params(in_million=True)

        optimizer = oto.hesso(
            variant='sgd',
            lr=0.1
        )
        oto.random_set_zero_groups()
        # YOLOv5 has some trouble to directly load torch model
        oto.construct_subnet(
            out_dir=OUT_DIR,
            ckpt_format='onnx'
        )

        full_sess = ort.InferenceSession(oto.full_group_sparse_model_path)
        full_output = full_sess.run(None, {'onnx::Cast_0': dummy_input.numpy()})
        compressed_sess = ort.InferenceSession(oto.compressed_model_path)
        compressed_output = compressed_sess.run(None, {'onnx::Cast_0': dummy_input.numpy()})

        max_output_diff = np.max(np.abs(full_output[0] - compressed_output[0]))
        print("Maximum output difference : ", max_output_diff.item())
        self.assertLessEqual(max_output_diff, 1e-3)

        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model        : ", full_model_size.st_size / (1024 ** 3), "GBs")
        print("Size of compress model    : ", compressed_model_size.st_size / (1024 ** 3), "GBs")

        # Compute FLOP and param for pruned model after oto.construct_subnet()
        pruned_flops = oto.compute_flops(in_million=True)['total']
        pruned_num_params = oto.compute_num_params(in_million=True)

        print("FLOP  reduction (%)       : ", 1.0 - pruned_flops / full_flops)
        print("Param reduction (%)       : ", 1.0 - pruned_num_params / full_num_params)