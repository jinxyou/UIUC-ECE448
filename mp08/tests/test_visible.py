import unittest
import traceback
from gradescope_utils.autograder_utils.decorators import weight, visibility, partial_credit

try:
    from submitted import baseline, viterbi
except Exception as e:
    print(e)
    print(traceback.format_exc())

import time
import utils

def test_synthetic():
    print('start synthetic test')
    train_set = utils.load_dataset('data/synthetic_training.txt')
    test_set = utils.load_dataset('data/synthetic_dev.txt')
    algorithm = viterbi
    try: 
        students_answer = algorithm(train_set, utils.strip_tags(test_set))
        accuracy, _, _ = utils.evaluate_accuracies(students_answer, test_set)
        if accuracy < 0.3:
            print('Synthetic test: there seems to be major bugs in your Viterbi algorithm')
            print('penalty multiplier: 0.2')
            return 0.2
        elif 0.3 <= accuracy < 0.9:
            print('Synthetic test: did you backtrace properly in your Viterbi algorithm?')
            print('penalty multiplier: 0.8')
            return 0.8
        elif accuracy >= 0.9:
            print('Synthetic test: passed!')
            return 1.0
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return 0.0

class TestMP4(unittest.TestCase):

    @weight(20)
    @visibility('visible')
    def test_brown_baseline(self):
        """test baseline on Brown"""
        train_set = utils.load_dataset('data/brown-training.txt')
        test_set = utils.load_dataset('data/brown-test.txt')
        algorithm = baseline
        name = algorithm.__name__
        max_time_spend = 60
        min_accuracy = 0.935
        min_multi_tag_accuracy = 0.900
        min_unseen_words_accuracy = 0.670
        try:
            _, time_spend, accuracy, multi_tag_accuracy, unseen_words_accuracy = runner(algorithm, train_set, test_set)
            self.assertLessEqual(time_spend, max_time_spend, "The {0} should run in less than {1} secs.".format(name, max_time_spend))
            self.assertGreaterEqual(accuracy, min_accuracy, "The {0} accuracy should be at least {1}.".format(name, min_accuracy))
            self.assertGreaterEqual(multi_tag_accuracy, min_multi_tag_accuracy, "The {0} multi-tag accuracy should be at least {1}.".format(name, min_multi_tag_accuracy))
            self.assertGreaterEqual(unseen_words_accuracy, min_unseen_words_accuracy, "The {0} unseen word accuracy should be at least {1}.".format(name, min_unseen_words_accuracy))
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.assertTrue(False, 'Error in baseline on Brown')


    @partial_credit(30)
    @visibility('visible')
    def test_brown_viterbi(self, set_score=None):
        """test viterbi on brown"""
        penalty = test_synthetic()

        train_set = utils.load_dataset('data/brown-training.txt')
        test_set = utils.load_dataset('data/brown-test.txt')
        algorithm = viterbi
        name = algorithm.__name__
        max_time_spend = 60
        level_1 = [0.87,0.87,0.17] # overall_accuracy, multi_tag_accuracy, unseen_words_accuracy
        level_2 = [0.90,0.90,0.19]
        level_3 = [0.925,0.925,0.20]
        try:
            _, time_spend, accuracy, multi_tag_accuracy, unseen_words_accuracy = runner(algorithm, train_set, test_set)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            set_score(0)
            self.assertTrue(False, 'Error in Viterbi 1 on Brown')    
            return
        #self.assertGreaterEqual(time_spend, max_time_spend, "Typically {0} should run in greater than {1} secs.".format(name, max_time_spend))
        total_score = 0
        if accuracy >= level_1[0] and multi_tag_accuracy >= level_1[1] and unseen_words_accuracy >= level_1[2]:
            total_score += 10
            print("+10 points for accuracy, multi_tag_accuracy, unseen_words_accuracy above {0} respectively.".format(level_1))
        else:
            print("The accuracy, multi_tag_accuracy, unseen_words_accuracy should be at least {0} respectively for 5 points.".format(level_1))
        
        if accuracy >= level_2[0] and multi_tag_accuracy >= level_2[1] and unseen_words_accuracy >= level_2[2]:
            total_score += 10
            print("+10 points for accuracy, multi_tag_accuracy, unseen_words_accuracy above {0} respectively.".format(level_2))
        else:
            print("The accuracy, multi_tag_accuracy, unseen_words_accuracy should be at least {0} respectively for 15 points.".format(level_2))
        
        if accuracy >= level_3[0] and multi_tag_accuracy >= level_3[1] and unseen_words_accuracy >= level_3[2]:
            total_score += 10
            print("+10 points for accuracy, multi_tag_accuracy, unseen_words_accuracy above {0} respectively.".format(level_3))
        else:
            print("The accuracy, multi_tag_accuracy, unseen_words_accuracy should be at least {0} respectively for 20 points.".format(level_3))
        
        # Viterbi 1 should not have good performance on the unseen word
        '''
        if unseen_words_accuracy >= 0.4:
            print("The accuracy for unseen words should not be too high for viterbi 1 algorithm.")
            set_score(0)
        else:
        '''
        total_score *= penalty
        set_score(total_score)

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


