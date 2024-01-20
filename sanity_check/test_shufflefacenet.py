import torch
from only_train_once import OTO
from backends import ShuffleFaceNet
import unittest
import os

OUT_DIR = './cache'

class TestShuffleFaceNet(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 112, 112)):
        model = ShuffleFaceNet()
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)

        oto = OTO(model, dummy_input)
        oto.mark_unprunable_by_node_ids(
            [
                'node-407', 'node-419', 'node-451', 'node-483', 'node-515', \
                'node-526', 'node-528', 'node-540', 'node-572', 'node-604', \
                'node-636', 'node-668', 'node-700', 'node-732', 'node-764', \
                'node-775', 'node-777', 'node-789', 'node-821', 'node-853', \
                'node-885'
            ]
        )
        oto.visualize(view=False, out_dir=OUT_DIR)

        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
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

        