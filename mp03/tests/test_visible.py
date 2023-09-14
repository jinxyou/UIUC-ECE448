import unittest, json, reader, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        with open('solution.json') as f:
            self.solution = json.load(f)
            
    @weight(20)
    def test_Kneighbors(self):
        ref1 = np.array(self.solution['neighbors'])
        ref2 = np.array(self.solution['labels'])
        train_images, train_labels, dev_images,dev_labels = reader.load_dataset('mp3_data', extra=True)
        neighbors, labels = submitted.k_nearest_neighbors(train_images[0], train_images, train_labels, k=2)
        M,N = ref1.shape[0],ref1.shape[1]
        K = ref2.shape[0]
        self.assertEqual((M,N), (neighbors.shape[0],neighbors.shape[1]),
                         'neighbors should be a 2-dimensional array with shape (%d,%d)'%(M,N))
        self.assertEqual(K, len(labels),
                    'labels should be a 1-dimensional array with length %d'%(K))
        for m in range(M):
            for n in range(N):
                self.assertAlmostEqual(ref1[m,n], neighbors[m,n], places=2,
                                       msg='''
                                       neighbors[%d,%d] should be %g, not %g
                                       '''%(m,n,ref1[m,n],neighbors[m,n]))
        for n in range(K):
            self.assertAlmostEqual(ref2[n], labels[n], places=2,
                                    msg='''
                                    neighbors[%d] should be %g, not %g
                                    '''%(n,ref2[n],labels[n]))

    @weight(15)
    def test_ClassifyDev(self):
        ref1 = np.array(self.solution['y_hats'])
        ref2 = np.array(self.solution['scores'])
        train_images, train_labels, dev_images,dev_labels = reader.load_dataset('mp3_data', extra=True)
        neighbors, labels = submitted.k_nearest_neighbors(train_images[0], train_images, train_labels, k=2)
        y_hats,scores = submitted.classify_devset(dev_images, train_images, train_labels, k=2)
        M,N = ref1.shape[0],ref2.shape[0]
        self.assertEqual(M, len(y_hats),
                         'y_hats should be a 1-dimensional array with length %d'%(M))
        self.assertEqual(N, len(scores),
                    'scores should be a 1-dimensional array with length %d'%(N))
        for m in range(M):
            self.assertAlmostEqual(ref1[m], y_hats[m], places=2,
                                    msg='''
                                    y_hats[%d] should be %g, not %g
                                    '''%(m,ref1[m],y_hats[m]))
        for n in range(N):
            self.assertAlmostEqual(ref2[n], scores[n], places=2,
                                    msg='''
                                    scores[%d] should be %g, not %g
                                    '''%(n,ref2[n],scores[n]))

    @weight(15)
    def test_Confusions(self):
        ref = np.array(self.solution['confusions'])
        ref1 = self.solution['accuracy']
        ref2 = self.solution['f1']
        train_images, train_labels, dev_images,dev_labels = reader.load_dataset('mp3_data', extra=True)
        y_hats,scores = np.array(self.solution['y_hats']),np.array(self.solution['scores'])
        confusions, accuracy, f1 = submitted.confusion_matrix(y_hats, dev_labels)
        M,N= ref.shape[0],ref.shape[1]
        self.assertEqual(M, confusions.shape[0],
                         'confusions should be a 1-dimensional array with length %d'%(M))
        self.assertAlmostEqual(ref1, accuracy, places=2,
                               msg='''accuracy should be %g, not %g'''%(ref1, accuracy))
        self.assertAlmostEqual(ref2, f1, places=2,
                               msg='''f1 should be %g, not %g'''%(ref2, f1))
        for m in range(M):
            for n in range(N):
                self.assertAlmostEqual(ref[m,n], confusions[m,n], places=2,
                                       msg='''
                                       confusions[%d,%d] should be %g, not %g
                                       '''%(m,n,ref[m,n],confusions[m,n]))


        
