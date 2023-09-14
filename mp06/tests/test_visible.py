import unittest, json, os
from gradescope_utils.autograder_utils.decorators import weight
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import submitted
from chess.lib.utils import encode, decode, initBoardVars
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

EXAMPLESDIR = 'grading_examples'

class nonrandomChoice():
    def __init__(self):
        self.i = 0
    def __call__(self, x):
        self.i = (self.i + 1) % len(x)
        return(x[self.i])

def convertMoves(moves):
    side, board, flags = initBoardVars()

    for fro, to, promote in map(decode, moves):
        side, board, flags = makeMove(side, board, fro, to, flags, promote)

    return side, board, flags

class load_game():
    def __init__(self,game):
        game_file = os.path.join("res", "savedGames", 'game%d.txt'%(game))
        with open(game_file, 'r') as f:
            lines = f.readlines()
        movestr = lines[2]
        moves = movestr.split()
        self.side, self.board, self.flags = convertMoves(moves)

def recursive_tree_test(testcase, label, ref, hyp, codelist):
    msg = label+' moveTree'+codelist+' should be dict, not '+str(type(hyp))
    testcase.assertIsInstance(hyp, dict, msg)
    for k in ref.keys():
        msg = label+' moveTree'+codelist+' should contain '+k+' but does not'
        testcase.assertIn(k, hyp, msg)
    for k in hyp.keys():
        msg = label+' moveTree'+codelist+' contains '+k+' but, that should have been pruned away.'
        testcase.assertIn(k, ref, msg)
        recursive_tree_test(testcase, label, ref[k], hyp[k], codelist+'['+k+']')

def general_test(testcase, app, label, searchfunc, depth, breadth=None):
    if breadth:
        chooser = nonrandomChoice()
        hypVal, hypList, hypTree = searchfunc(app.side, app.board, app.flags, depth, breadth, chooser)
    else:
        hypVal, hypList, hypTree = searchfunc(app.side, app.board, app.flags, depth)
    with open(os.path.join(EXAMPLESDIR, label+'.json')) as f:
        refVal = float(f.readline())
        refList = json.loads(f.readline())
        refTree = json.loads(f.readline())
    msg = '%s should produce value == %s not %s'%(label,refVal, hypVal)
    testcase.assertEqual(refVal, hypVal, msg)
    if breadth:  # For stochastic search, test only the first move
        msg = '%s moveList should be a non-empty list, not %s'%(label,hypList)
        testcase.assertIsInstance(hypList, list, msg)
        testcase.assertGreater(len(hypList), 0, msg)
        msg = '%s should produce moveList[0] == %s not %s'%(label,refList[0], hypList[0])
        testcase.assertEqual(refList[0], hypList[0], msg)
    else:  # Don't test the moveList for stochastic search
        msg = '%s should produce moveList==%s not %s'%(label,refList, hypList)
        testcase.assertEqual(refList, hypList, msg)
    recursive_tree_test(testcase, label, refTree, hypTree, '')
    
class grading_tests(unittest.TestCase):
    @weight(8)
    def test_minimax01(self):
        general_test(self, load_game(0), 'minimax_game%d_depth%d'%(0,1), submitted.minimax, 1)
    @weight(7)
    def test_minimax02(self):
        general_test(self, load_game(0), 'minimax_game%d_depth%d'%(0,2), submitted.minimax, 2)
    @weight(7)
    def test_alphabeta02(self):
        general_test(self, load_game(0), 'alphabeta_game%d_depth%d'%(0,2), submitted.alphabeta, 2)

    @weight(7)
    def test_minimax11(self):
        general_test(self, load_game(1), 'minimax_game%d_depth%d'%(1,1), submitted.minimax, 1)
    @weight(7)
    def test_minimax12(self):
        general_test(self, load_game(1), 'minimax_game%d_depth%d'%(1,2), submitted.minimax, 2)
    @weight(7)
    def test_alphabeta12(self):
        general_test(self, load_game(1), 'alphabeta_game%d_depth%d'%(1,2), submitted.alphabeta, 2)
    @weight(7)
    def test_alphabeta13(self):
        general_test(self, load_game(1), 'alphabeta_game%d_depth%d'%(1,3), submitted.alphabeta, 3)
        
