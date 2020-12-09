import numpy as np
import random

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIDE = 3
BOARD_SIZE = BOARD_SIDE * BOARD_SIDE

class BaseState:
    def __init__(self, p1, p2, lose_reward=-1, win_reward=1, p1_tie_reward=0.1, p2_tie_reward=0.5):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1
        self.results = []
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.p1_tie_reward = p1_tie_reward
        self.p2_tie_reward = p2_tie_reward

        # Metric tracking
        self.p1_wins = 0
        self.p2_wins = 0
        self.tie = 0
        self.games = 0


    def reset_metrics(self):
        self.p1_wins = 0
        self.p2_wins = 0
        self.tie = 0
        self.games = 0

    # The default way to get hash of board
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    # only when game ends
    def giveReward(self):
        result = self.winner()
        self.games += 1
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(self.win_reward)
            self.p2.feedReward(self.lose_reward)
            self.p1_wins += 1
        elif result == -1:
            self.p1.feedReward(self.lose_reward)
            self.p2.feedReward(self.win_reward)
            self.p2_wins += 1
        else:
            self.p1.feedReward(self.p1_tie_reward)
            self.p2.feedReward(self.p2_tie_reward)
            self.tie += 1

    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1



