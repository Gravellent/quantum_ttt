import pickle
import numpy as np
import copy
from model import *
from collections import defaultdict
from torch import optim, nn

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIDE = 3
BOARD_SIZE = BOARD_SIDE * BOARD_SIDE

class BasePlayer:
    def __init__(self, name, exp_rate=0.3, player_symbol=1, update_method='sarsa'):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value
        self.player_symbol = player_symbol
        self.update_method = update_method
        self.is_eval = False
        self.model = None

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS*BOARD_ROWS))
        return boardHash

    def addState(self, state):
        pass

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

    def forget(self):
        self.model = None
        self.states_value = {}

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

