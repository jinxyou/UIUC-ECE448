import os,sys,argparse
import random
import pygame
import chess.lib
import submitted

################################################################################################
# Main chess-playing application.
# Mark Hasegawa-Johnson, March 2021,
# based on PyChess/chess/mysingleplayer.py.
#
class Application():
    def __init__(self, players, depths, breadths, movestr="", heuristic=None):
        self.moves = movestr.split()
        self.side, self.board, self.flags = chess.lib.convertMoves(self.moves)

        # mp5 player0 corresponds to "side False"==white==MAX in PyChess
        self.player = players
        self.depth = depths
        self.breadth = breadths
        self.sel = [0,0]
        if heuristic:
            self.heuristic = heuristic

        self.buttons = {
            'quit' : pygame.Rect(460,0,40,50),
            'saveGame' : pygame.Rect(350,460,150,30),
            'undo' : pygame.Rect(0,0,80,50)
        }

    def close(self):
        pygame.quit()
        quit()

    def makemove(self, fro, to):
        '''
        Check whether move fro -> to results in pawn promotion.
        Animate the move on the game baord.
        Morph self.side, self.board, and self.flags in response to the move.
        Append the move to the list self.moves.
        '''
        promote = chess.lib.getPromote(self.win, self.side, self.board, fro, to)
        if not chess.lib.core.getType(self.side, self.board, fro):
            raise RunTimeError('Player %d has no piece at position %d, %d'%(int(self.side),fro[0],fro[1]))
        chess.lib.animate(self.win, int(self.side), self.board, fro, to, self.prefs,
                          self.player[True]=='human')
        self.side, self.board, self.flags = chess.lib.makeMove(
            self.side,  self.board, fro, to, self.flags, promote)
        self.moves.append(chess.lib.encode(fro, to, promote))

    def heuristic_move(side, board, flags):
        'Use a provided heuristic function to choose a move'
        return(self.heuristic(side, board, flags))
        
    def run(self):
        pygame.init()
        if pygame.version.vernum[0] >= 2:
            self.win = pygame.display.set_mode((500, 500), pygame.SCALED)
        else:
            self.win = pygame.display.set_mode((500, 500))

        self.prefs = { 'flip' : True, 'allow_undo' : True, 'show_moves' : True }

        clock = pygame.time.Clock()
        chess.lib.start(self.win, self.prefs)

        # Continue until there is no move left for the current player
        while not chess.lib.isEnd(self.side, self.board, self.flags):
            clock.tick(25)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if self.buttons['quit'].collidepoint(x,y):
                        self.close()
                    elif self.buttons['saveGame'].collidepoint(x,y):
                        if chess.lib.prompt(self.win, chess.lib.saveGame(self.moves, "mp5", player)):
                            self.close()
                    elif self.buttons['undo'].collidepoint(x,y):
                        if self.player[self.side]=='human':
                            self.moves = chess.lib.undo(self.moves, 2)
                        else:
                            self.moves = chess.lib.undo(self.moves)
                        self.side, self.board, self.flags = chess.lib.convertMoves(self.moves)

                    elif self.player[self.side] == 'human' and 50 < x < 450 and 50 < y < 450:
                        x, y = x // 50, y // 50         # convert x and y to chess coords
                        if self.prefs["flip"] and self.player[True]=='human':  # if board is flipped
                            x, y = 9 - x, 9 - y
                        fro = self.sel
                        self.sel = to = [x, y]
                        if chess.lib.isValidMove(self.side, self.board, self.flags, fro, to):
                            self.makemove(fro, to)

            if self.player[self.side] != 'human':
                self.sel = [0,0]
                if self.player[self.side] == 'random':
                    value, moveList, moveTree = submitted.random(
                        self.side, self.board, self.flags, random.choice)
                elif self.player[self.side] == 'minimax':
                    value, moveList, moveTree = submitted.minimax(
                        self.side, self.board, self.flags, self.depth[self.side])
                elif self.player[self.side] == 'heuristic':
                    value, moveList, moveTree = heuristic_move(
                        self.side, self.board, self.flags)
                elif self.player[self.side] == 'alphabeta':
                    value, moveList, moveTree = submitted.alphabeta(
                        self.side, self.board, self.flags, self.depth[self.side])
                elif self.player[self.side] == 'stochastic':
                    value, moveList, moveTree = submitted.stochastic(
                        self.side, self.board, self.flags, self.depth[self.side],
                        self.breadth[self.side], random.choice)
                elif self.player[self.side] == 'extracredit':
                    value, moveList, moveTree = submitted.stochastic(
                        self.side, self.board, self.flags, self.depth[self.side],
                        self.breadth[self.side], random.choice)
                self.makemove(moveList[0][0], moveList[0][1])
            
            chess.lib.showScreen(self.win, self.side, self.board, self.flags, self.sel,
                                 self.prefs, self.player[True]=='human')

        # Now that the game is done, continue showing the screen until the user clicks quit
        while True:
            clock.tick(25)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if self.buttons['quit'].collidepoint(x,y):
                        self.close()
                    elif self.buttons['saveGame'].collidepoint(x,y):
                        if chess.lib.prompt(self.win, chess.lib.saveGame(self.moves, "mp5", self.side)):
                            self.close()
                    elif self.buttons['undo'].collidepoint(x,y):
                        self.moves = chess.lib.undo(self.moves)
                        self.side, self.board, self.flags = chess.lib.convertMoves(self.moves)
                        chess.lib.showScreen(self.win, self.side, self.board, self.flags, self.sel,
                                             self.prefs, self.player[True]=='human')

################################################################################################
# Command line arguments
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description     = 'CS440 MP5 Chess', 
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--player0', default = 'human',
                        choices = ('random', 'human', 'minimax', 'alphabeta', 'stochastic'),
                        help = 'Is player 0 a human, a random player, or some type of AI?')
    parser.add_argument('--player1', default = 'random',
                        choices = ('random', 'human', 'minimax', 'alphabeta', 'stochastic'),
                        help = 'Is player 1 a human, a random player, or some type of AI?')
    parser.add_argument('--depth0', type=int, default=2,
                        help = 'Depth to which player 0 should search, if player 0 is an AI.')
    parser.add_argument('--depth1', type=int, default=2,
                        help = 'Depth to which player 1 should search, if player 1 is an AI.')
    parser.add_argument('--breadth0', type=int, default=2,
                        help = 'Breadth to which player 0 should search, if player 0 is stochastic.')
    parser.add_argument('--breadth1', type=int, default=2,
                        help = 'Breadth to which player 1 should search, if player 1 is stochastic.')
    parser.add_argument('--loadgame', type=str, default=None,
                        help = 'Load a saved game from res/savedGames')
                        

    args   = parser.parse_args()

    # Load a previous game, if requested to do so
    movestr = ""
    if args.loadgame:
        name = os.path.join("res", "savedGames", args.loadgame)
        if os.path.exists(name):
            with open(name, "r") as file:
                lines = file.readlines()
            movestr = lines[2]

    # Create and run the application
    application = Application([args.player0, args.player1], [args.depth0, args.depth1],
                              [args.breadth0, args.breadth1], movestr)
    application.run()
