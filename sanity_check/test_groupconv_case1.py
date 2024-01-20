import torch
from only_train_once import OTO
from backends import DemoNetGroupConvCase1
import unittest
import os

OUT_DIR = './cache'

class TestGroupConvCase1(unittest.TestCase):
    def test_sanity(
            self, 
            dummy_input=(
                torch.rand(1, 3, 512, 512),
                torch.rand(1, 3, 512, 512),
                torch.rand(1, 384, 16, 16),
                torch.rand(1, 64, 16, 16)
            )
        ):
        affine = True
        norm_type = 'in'
        model = DemoNetGroupConvCase1(norm_type=norm_type, affine=affine)
        oto = OTO(model, dummy_input)
        unprunable_param_names = [
            'conv_1.conv1.weight',
            'conv_5.conv2.weight',
            'conv_6.conv1.weight'
        ]
        oto.mark_unprunable_by_param_names(param_names=unprunable_param_names)

        oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)
        # For test FLOP and param reductions. 
        full_flops = oto.compute_flops(in_million=True)['total']
        full_num_params = oto.compute_num_params(in_million=True)

        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(*dummy_input)
        compressed_output = compressed_model(*dummy_input)

        max_output_diff = torch.max(torch.abs(full_output[0] - compressed_output[0]))
        print("Maximum output difference " + str(max_output_diff.item()))
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
        print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")
        self.assertLessEqual(max_output_diff, 1e-3)

        # For test FLOP and param reductions. 
        oto_compressed = OTO(compressed_model, dummy_input)
        compressed_flops = oto_compressed.compute_flops(in_million=True)['total']
        compressed_num_params = oto_compressed.compute_num_params(in_million=True)

        print("FLOP  reduction (%)    : ", 1.0 - compressed_flops / full_flops)
        print("Param reduction (%)    : ", 1.0 - compressed_num_params / full_num_params)