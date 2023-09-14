import argparse
import numpy as np
import submitted
import torch
import traceback
import unittest
from gradescope_utils.autograder_utils.decorators import partial_credit, weight


# I presume Python unittest will pass to the tests the same arguments we feed to the command-line interface,
# but I am not completely sure if this is the case. Anyway, it seems to work.


parser = argparse.ArgumentParser(description="CS440/ECE448 MP: Perception")

parser.add_argument(
    "--epochs",
    dest="epochs",
    type=int,
    default=1,
    help="Training Epochs: default 1",
)
parser.add_argument(
    "--batch",
    dest="batch",
    type=int,
    default=64,
    help="Batch size: default 64",
)
parser.add_argument(
    "--seed", dest="seed", type=int, default=42, help="seed source for randomness"
)
parser.add_argument(
    "-j", "--json", action="store_true", help="""Results in Gradescope JSON format."""
)

args = parser.parse_args()


class Test(unittest.TestCase):
    @weight(15)
    def test_dataset(self):
        try:
            test_set = submitted.build_dataset(["cifar10_batches/test_batch"])
            self.assertEquals(len(test_set), 8000)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.assertTrue(False, "Test dataset not correctly built.")

    @weight(15)
    def test_dataloader(self):
        try:
            batch_size = 64
            test_set = submitted.build_dataset(["cifar10_batches/test_batch"])
            test_dataloader = submitted.build_dataloader(test_set, loader_params={"batch_size": batch_size, "shuffle": True})
            num_batches = len(test_set) // batch_size
            if len(test_set) % batch_size != 0:
                num_batches += 1
            self.assertEquals(len(test_dataloader), num_batches)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.assertTrue(False, "Test dataloader not correctly built.")
            
