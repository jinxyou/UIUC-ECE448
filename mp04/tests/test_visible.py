import argparse
import numpy as np
import reader as r
import submitted0
import torch
import traceback
import unittest
from gradescope_utils.autograder_utils.decorators import partial_credit, weight


# I presume Python unittest will pass to the tests the same arguments we feed to the command-line interface,
# but I am not completely sure if this is the case. Anyway, it seems to work.


parser = argparse.ArgumentParser(description="CS440/ECE448 MP: Neural Nets and PyTorch")

parser.add_argument(
    "--epochs",
    dest="epochs",
    type=int,
    default=50,
    help="Training Epochs: default 50",
)
parser.add_argument(
    "--batch",
    dest="batch",
    type=int,
    default=100,
    help="Batch size: default 100",
)
parser.add_argument(
    "--seed", dest="seed", type=int, default=42, help="seed source for randomness"
)
parser.add_argument(
    "-j", "--json", action="store_true", help="""Results in Gradescope JSON format."""
)

args = parser.parse_args()


class Test(unittest.TestCase):
    def setUp(self):
        # Load data
        r.init_seeds(args.seed)
        train_set, train_labels, test_set, test_labels = r.Load_dataset("data/mp_data")
        train_set, test_set = r.Preprocess(train_set, test_set)
        train_loader, test_loader = r.Get_DataLoaders(
            train_set, train_labels, test_set, test_labels, args.batch
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_set = test_set
        self.test_labels = test_labels

    @weight(10)
    def test_loss_fn(self):
        try:
            _, loss_fn, _ = submitted.fit(self.train_loader, self.test_loader, 0)
            self.assertIsInstance(loss_fn, torch.nn.modules.loss._Loss)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.assertTrue(False, "Error in loss function. Run locally to debug.")

    @weight(15)
    def test_optimizer(self):
        try:
            _, _, optimizer = submitted.fit(self.train_loader, self.test_loader, 0)
            self.assertIsInstance(optimizer, torch.optim.Optimizer)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.assertTrue(False, "Error in optimizer. Run locally to debug.")

    @partial_credit(40)
    def test_accuracy(self, set_score=None):
        # Train
        try:
            model, _, _ = submitted.fit(
                self.train_loader, self.test_loader, args.epochs
            )
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.assertTrue(
                False, "Error in neural net implementation. Run locally to debug."
            )
        # Predict
        pred_values = model(self.test_set)  # Predicted value of the testing set
        pred_values = pred_values.detach().numpy()
        pred_labels = np.argmax(pred_values, axis=1)  # Predicted labels
        # Error handling
        self.assertEquals(
            len(pred_labels),
            len(self.test_labels),
            "Incorrect size of predicted labels.",
        )
        num_parameters = sum([np.prod(w.shape) for w in model.parameters()])
        upper_threshold = 10000
        lower_threshold = 1000000
        print("Total number of network parameters: ", num_parameters)
        self.assertLess(
            num_parameters,
            lower_threshold,
            "Your network is way too large with "
            + str(num_parameters)
            + " parameters. The upper limit is "
            + str(lower_threshold)
            + "!",
        )
        self.assertGreater(
            num_parameters,
            upper_threshold,
            "Your network is suspiciously compact. Have you implemented something other than a neural network?"
            + " Or perhaps the number of hidden neurons is too small. Neural nets usually have over "
            + str(upper_threshold)
            + " parameters!",
        )
        # Accuracy test
        accuracy, conf_mat = r.compute_accuracies(pred_labels, self.test_labels)
        print("\n Accuracy:", accuracy)
        print("\nConfusion Matrix = \n {}".format(conf_mat))
        # Compute score
        score = 0
        for threshold in [0.15, 0.25, 0.48, 0.55]:
            if accuracy >= threshold:
                score += 5
                print("+5 points for accuracy above", str(threshold))
            else:
                break
        for threshold in [0.57, 0.61]:
            if accuracy >= threshold:
                score += 10
                print("+10 points for accuracy above", str(threshold))
            else:
                break
        if score != 40:
            print("Accuracy must be above 0.61")
        set_score(score)
