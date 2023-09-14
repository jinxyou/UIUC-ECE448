import unittest
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

def post_processor(json_file):
    for test in json_file['tests']:
        test['output'] = test['output'].replace('\n', '; ')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover('tests')
    JSONTestRunner(visibility='visible', post_processor=post_processor).run(suite)
