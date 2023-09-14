"""
This file is a part of My-PyChess application.
In this file, we define heuristic constants required for the python chess
engine.
"""

pawnEvalWhite = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
    (2.0, 2.0, 3.0, 5.0, 5.0, 3.0, 2.0, 2.0),
    (0.5, 0.5, 1.0, 2.5, 2.5, 1.0, 0.5, 0.5),
    (0.0, 0.0, 0.5, 2.0, 2.0, 0.5, 0.0, 0.0),
    (0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5),
    (0.5, 1.0, 0.5, -2.0, -2.0, 0.5, 1.0, 0.5),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)

pawnEvalBlack = tuple(reversed(pawnEvalWhite))

knightEval = (
    (-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0),
    (-4.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, -4.0),
    (-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0),
    (-3.0, 0.5, 1.5, 2.0, 2.0, 1.5, 0.5, -3.0),
    (-3.0, 0.0, 1.5, 2.0, 2.0, 1.5, 0.0, -3.0),
    (-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0),
    (-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0),
    (-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0),
)

bishopEvalWhite = (
    (-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0),
    (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -1.0),
    (-1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, -1.0),
    (-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0),
    (-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0),
    (-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0),
    (-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0),
)

bishopEvalBlack = tuple(reversed(bishopEvalWhite))

rookEvalWhite = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5),
    (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
    (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
    (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
    (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
    (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
    (0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0),
)

rookEvalBlack = tuple(reversed(rookEvalWhite))

queenEval = (
    (-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0),
    (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0),
    (-0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5),
    (0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5),
    (-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0),
    (-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0),
    (-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0),
)

kingEvalWhite = (
    (-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0),
    (-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0),
    (-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0),
    (-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0),
    (-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0),
    (-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0),
    (2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0),
    (2.0, 3.0, 3.0, 0.0, 0.0, 1.0, 3.0, 2.0),
)

kingEvalBlack = tuple(reversed(kingEvalWhite))

# This is a rudementary and simple evaluative function for a given state of
# board. It gives each piece a value based on its position on the board,
# returns a numeric representation of the board
def evaluate(board):
    score = 0
    for x, y, piece in board[0]:
        if piece == "p":
            score += 1 + pawnEvalWhite[y - 1][x - 1]
        elif piece == "b":
            score += 9 + bishopEvalWhite[y - 1][x - 1]
        elif piece == "n":
            score += 9 + knightEval[y - 1][x - 1]
        elif piece == "r":
            score += 14 + rookEvalWhite[y - 1][x - 1]
        elif piece == "q":
            score += 25 + queenEval[y - 1][x - 1]
        elif piece == "k":
            score += 200 + kingEvalWhite[y - 1][x - 1]

    for x, y, piece in board[1]:
        if piece == "p":
            score -= 1 + pawnEvalBlack[y - 1][x - 1]
        elif piece == "b":
            score -= 9 + bishopEvalBlack[y - 1][x - 1]
        elif piece == "n":
            score -= 9 + knightEval[y - 1][x - 1]
        elif piece == "r":
            score -= 14 + rookEvalBlack[y - 1][x - 1]
        elif piece == "q":
            score -= 25 + queenEval[y - 1][x - 1]
        elif piece == "k":
            score -= 200 + kingEvalBlack[y - 1][x - 1]

    return score

