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
        oto.mark_unprunable_by_node_ids(
            # ['node-229', 'node-329', 'node-443', 'node-553']
            ['node-229', 'node-581', 'node-471', 'node-359']
        )
        oto.visualize(view=False, out_dir=OUT_DIR)
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
        print("Maximum output difference " + str(max_output_diff.item()))
        self.assertLessEqual(max_output_diff, 1e-3)

        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
        print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")