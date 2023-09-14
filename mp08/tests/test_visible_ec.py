import unittest
import traceback
from gradescope_utils.autograder_utils.decorators import weight, visibility, partial_credit

try:
    from submitted import viterbi_ec
except Exception as e:
    print(e)
    print(traceback.format_exc())

import time
import utils

class TestMP4(unittest.TestCase):

    @partial_credit(5)
    @visibility('visible')
    def test_brown_viterbi_ec(self, set_score=None):
        """test viterbi_ec on Brown"""
        train_set = utils.load_dataset('data/brown-training.txt')
        test_set = utils.load_dataset('data/brown-test.txt')
        algorithm = viterbi_ec
        name = algorithm.__name__
        level_1 = [0.94,0.92,0.50]
        level_2 = [0.94,0.92,0.60]
        level_3 = [0.95,0.932,0.65] #overall_accuracy, multi_tag_accuracy, unseen_words_accuracy
        try:
            _, time_spend, accuracy, multi_tag_accuracy, unseen_words_accuracy = runner(algorithm, train_set, test_set)
            total_score = 0
            if accuracy >= level_1[0] and multi_tag_accuracy >= level_1[1] and unseen_words_accuracy >= level_1[2]:
                total_score += 1.5
                print("+1.5 points for accuracy, multi_tag_accuracy, unseen_words_accuracy above {0} respectively.".format(level_1))
            else:
                print("The accuracy, multi_tag_accuracy, unseen_words_accuracy should be at least {0} respectively for 5 points.".format(level_1))
            
            if accuracy >= level_2[0] and multi_tag_accuracy >= level_2[1] and unseen_words_accuracy >= level_2[2]:
                total_score += 1.5
                print("+1.5 points for accuracy, multi_tag_accuracy, unseen_words_accuracy above {0} respectively.".format(level_2))
            else:
                print("The accuracy, multi_tag_accuracy, unseen_words_accuracy should be at least {0} respectively for 10 points.".format(level_2))

            if accuracy >= level_3[0] and multi_tag_accuracy >= level_3[1] and unseen_words_accuracy >= level_3[2]:
                total_score += 2
                print("+2 points for accuracy, multi_tag_accuracy, unseen_words_accuracy above {0} respectively.".format(level_3))
            else:
                print("The accuracy, multi_tag_accuracy, unseen_words_accuracy should be at least {0} respectively for 15 points.".format(level_3))

            set_score(total_score)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            set_score(0)
            self.assertTrue(False, 'Error in Viterbi_ec on Brown')

def runner(algorithm, train_set, test_set):
    try:
        start_time = time.time()
        students_answer = algorithm(train_set, utils.strip_tags(test_set))
        time_spend = time.time() - start_time
        accuracy, _, _ = utils.evaluate_accuracies(students_answer, test_set)
        multi_tag_accuracy, unseen_words_accuracy, = utils.specialword_accuracies(train_set, students_answer, test_set)

        print("time spent: {0:.4f} sec".format(time_spend))
        print("accuracy: {0:.4f}".format(accuracy))
        print("multi-tag accuracy: {0:.4f}".format(multi_tag_accuracy))
        print("unseen word accuracy: {0:.4f}".format(unseen_words_accuracy))

        return algorithm.__name__, time_spend, accuracy, multi_tag_accuracy, unseen_words_accuracy

    except:
        traceback.print_exc()
        raise InterruptedError

