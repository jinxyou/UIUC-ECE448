import unittest, argparse
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the grade.')
    parser.add_argument('--gradescope',action='store_true',
                        help='''Instead of printing output to the terminal, write it to a 
                        JSON file in the format used on Gradescope.''')
    parser.add_argument('--profiler',action='store_true',
                        help='''Run profiler, to see how long each test is taking.''')
    args = parser.parse_args()

    # This line looks for unit tests in the directory tests
    suite = unittest.defaultTestLoader.discover('tests')

    # Now, use python's unittest facility to run the tests, in either gradescope or text format
    if args.gradescope:
        JSONTestRunner(visibility='visible').run(suite)
    elif args.profiler:
        import profile
        profile.run('result = unittest.TextTestRunner().run(suite)')
    else:
        result = unittest.TextTestRunner().run(suite)
