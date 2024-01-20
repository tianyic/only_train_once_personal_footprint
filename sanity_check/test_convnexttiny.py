import torch
from only_train_once import OTO
from backends import convnext_tiny
import unittest
import os

OUT_DIR = './cache'

class TestConvNextTiny(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 224, 224)):
        # layer_scale_init_value will disable a nn.Parameters gamma. 
        # The singleton parameter is not supported in OTOv3
        # Pretrained convnext has such parameters, thereby need to skip a few node groups.
        model = convnext_tiny(layer_scale_init_value=-1)
        # model = convnext_tiny(pretrained=True)
        oto = OTO(model, dummy_input)        
        
        # # For pretrained convnexttiny
        # oto.mark_unprunable_by_node_ids(['node-183', 'node-312', 'node-422', 'node-712'])

        oto.visualize(view=False, out_dir=OUT_DIR)
        # For test FLOP and param reductions. 
        full_flops = oto.compute_flops(in_million=True)['total']
        full_num_params = oto.compute_num_params(in_million=True)

        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(dummy_input)
        compressed_output = compressed_model(dummy_input)

        max_output_diff = torch.max(torch.abs(full_output - compressed_output))
        print("Maximum output difference " + str(max_output_diff.item()))
        # self.assertLessEqual(max_output_diff, 1e-4)
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
        print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")

        # For test FLOP and param reductions. 
        oto_compressed = OTO(compressed_model, dummy_input)
        compressed_flops = oto_compressed.compute_flops(in_million=True)['total']
        compressed_num_params = oto_compressed.compute_num_params(in_million=True)

        print("FLOP  reduction (%)    : ", 1.0 - compressed_flops / full_flops)
        print("Param reduction (%)    : ", 1.0 - compressed_num_params / full_num_params)