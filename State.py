import numpy as np
import random

BOARD_ROWS = 3
BOARD_COLS = 3

class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1
        self.results = []

    # get unique hash of current board state
    def getHash(self):
        #         print('state', self.board)
        #        board_str = ''
        #        for pos in self.board.reshape(BOARD_COLS*BOARD_ROWS):
        #            if pos == -1:
        #                board_str += '2'
        #            else:
        #                board_str += str(int(pos))
        #       self.boardHash = board_str
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        # col
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(diag_sum1, diag_sum2)
        if diag_sum == 3:
            self.isEnd = True
            return 1
        if diag_sum == -3:
            self.isEnd = True
            return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0 and self.collapsable() == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def collapsable(self):
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0.5 or self.board[i, j] == -0.5:
                    return 1
        return 0

    def collapse(self):
        index1 = np.argwhere(self.board == 0.5)
        index2 = np.argwhere(self.board == -0.5)
        seed1 = random.sample(range(0, len(index1)), int((len(index1) / 2)))
        seed2 = random.sample(range(0, len(index2)), int((len(index2) / 2)))

        for i1, j1 in enumerate(seed1):
            self.board[index1[j1][0], index1[j1][1]] = 1
        for i2, j2 in enumerate(seed2):
            self.board[index2[j2][0], index2[j2][1]] = -1
        for m in range(BOARD_ROWS):
            for n in range(BOARD_COLS):
                if self.board[m, n] != 1 and self.board[m, n] != -1:
                    self.board[m, n] = 0

        return self.board

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        #         print('avail', positions)
        return positions

    def updateState(self, position1, position2):
        self.board[position1] += self.playerSymbol * 0.5
        self.board[position2] += self.playerSymbol * 0.5
        #         print(len(self.availablePositions()))
        if len(self.availablePositions()) == 0 and self.collapsable() == 1:
            self.board = self.collapse()
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100, verbose=False):
        for i in range(rounds):
            #             if i%1000 == 0:
            #                 print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action1, p1_action2 = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and upate board state
                self.updateState(p1_action1, p1_action2)
                if verbose:
                    print('P1', self.board)

                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end
                #                 print('P1', self.board)

                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.results.append(win)
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action1, p2_action2 = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action1, p2_action2)
                    if verbose:
                        print('P2', self.board)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.results.append(win)
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

    # play with human
    def play2(self):
        while not self.isEnd:
            # Player 1
            positions = self.availablePositions()
            p1_action1, p1_action2 = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            # take action and upate board state
            self.updateState(p1_action1, p1_action2)
            self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.availablePositions()
                p2_action1, p2_action2 = self.p2.chooseAction(positions)

                self.updateState(p2_action1, p2_action2)
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'X'
                if self.board[i, j] == -1:
                    token = 'O'
                if self.board[i, j] == 0.5:
                    token = 'x'
                if self.board[i, j] == -0.5:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')