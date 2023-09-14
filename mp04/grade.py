import unittest, argparse
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

# Create a parser
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

def main():
    # Run tests
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover("tests")
    if args.json:
        JSONTestRunner(visibility="visible").run(suite)
    else:
        result = unittest.TextTestRunner().run(suite)

if __name__ == '__main__':
    main()
