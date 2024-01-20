import torch
from only_train_once import OTO
from backends import TinyUNet
import unittest
import os

OUT_DIR = './cache'

class TestTinyUnet(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 64, 64)):
        model = TinyUNet(device='cpu')
        dummy_input = (torch.randn(1, 3, 64, 64), torch.ones((1,)).long())
        oto = OTO(model, dummy_input)
        oto.visualize(view=False, out_dir=OUT_DIR)
        oto.random_set_zero_groups()
        oto.compress(out_dir=OUT_DIR)
        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(dummy_input)
        compressed_output = compressed_model(dummy_input)

        max_output_diff = torch.max(torch.abs(full_output - compressed_output))
        print("Maximum output difference " + str(max_output_diff.item()))
        self.assertLessEqual(max_output_diff, 1e-4)
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
        print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")

        