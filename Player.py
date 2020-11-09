import pickle
import numpy as np

BOARD_ROWS = 3
BOARD_COLS = 3

class Player:
    def __init__(self, name, exp_rate=0.3, player_symbol=1, update_method='sarsa'):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value
        self.player_symbol = player_symbol
        self.update_method = update_method

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS*BOARD_ROWS))
        return boardHash

    # get unique hash of current board state
    # def getHash(self, board):
    #     board_str = ''
    #     for pos in board.reshape(BOARD_COLS * BOARD_ROWS):
    #         if pos == -1:
    #             board_str += '2'
    #         else:
    #             board_str += str(int(pos))
    #     self.boardHash = board_str
    #     return self.boardHash

    def chooseAction(self, positions, current_board, symbol):
        # take random action
        idx1 = np.random.choice(len(positions))
        action1 = positions[idx1]
        idx2 = np.random.choice(len(positions))
        action2 = positions[idx2]

        if np.random.uniform(0, 1) > self.exp_rate:
            value_max = -999
            for p1 in positions:
                for p2 in positions:
                    next_board = current_board.copy()
                    next_board[p1] += symbol * 0.5
                    next_board[p2] += symbol * 0.5
                    next_boardHash = self.getHash(next_board)
                    value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(
                        next_boardHash)
                    # print("value", value)
                    if value >= value_max:
                        value_max = value
                        action1 = p1
                        action2 = p2

        # print("{} takes action {}".format(self.name, action))
        return action1, action2

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # Find the distance between states
    def is_next_state(self, old_state, new_state):
        #         print('Compare', old_state, new_state)
        diff = []
        for pos1, pos2 in zip(old_state, new_state):
            if pos1 == pos2:
                continue
            if pos1 == '0' and pos2 != '0':
                diff.append(pos2)
            else:
                return False
        #         print('diff', diff)
        return sorted(diff) == ['1', '2']

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        if self.update_method == 'sarsa':
            for st in reversed(self.states):
                if self.states_value.get(st) is None:
                    self.states_value[st] = 0
                self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
                reward = self.states_value[st]

        if self.update_method == 'expected_sarsa':
            #             print('ss', self.states, 'reward', reward)
            for st in reversed(self.states):
                if self.states_value.get(st) is None:
                    self.states_value[st] = 0
                possible_states = []
                if reward is None:
                    for next_st in self.states_value:
                        if self.is_next_state(st, next_st):
                            possible_states.append((next_st, self.states_value[next_st]))
                    reward = max([_[1] for _ in possible_states])
                #                     print(possible_states, reward)
                self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
                #                 print(self.states_value)
                reward = None

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

class RandomPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions, current_board, symbol):
        #    print(positions)
        return positions[np.random.choice(len(positions))], positions[np.random.choice(len(positions))]

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row1 = int(input("Input your action row:"))
            col1 = int(input("Input your action col:"))
            action1 = (row1, col1)
            row2 = int(input("Input your action row:"))
            col2 = int(input("Input your action col:"))
            action2 = (row2, col2)
            if action1 in positions and action2 in positions:
                return action1, action2

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass