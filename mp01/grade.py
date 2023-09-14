import unittest, argparse

parser = argparse.ArgumentParser(description='''run unit tests for MP.''')
parser.add_argument('-j','--json',action='store_true',help='''Results in Gradescope JSON format.''')
args = parser.parse_args()
suite = unittest.defaultTestLoader.discover('tests')
if args.json:
    from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner
    JSONTestRunner(visibility='visible').run(suite)
else:
    result = unittest.TextTestRunner().run(suite)
