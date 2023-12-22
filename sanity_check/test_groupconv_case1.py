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

        unprunable_node_ids = ['out-67', 'out-88', 'out-84']    
            
        for node_group in oto._graph.node_groups.values():
            for unprunable_node_id in unprunable_node_ids:
                if unprunable_node_id in node_group.id:
                    node_group.is_prunable = False

        oto.visualize(view=False, out_dir=OUT_DIR)
        
        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        torch.onnx.export(
            model,
            dummy_input,
            'sanity_check_demo_groupconv.onnx'
        )
        full_output = full_model(*dummy_input)
        compressed_output = compressed_model(*dummy_input)

        max_output_diff = torch.max(torch.abs(full_output[0] - compressed_output[0]))
        print("Maximum output difference " + str(max_output_diff.item()))
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
        print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")
        self.assertLessEqual(max_output_diff, 3.0)
