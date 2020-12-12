from Player.BasePlayer import *
from utils import *
import random

class QPlayer(BasePlayer):

    def __init__(self, name, memory_type="dict", capacity=10000):
        super().__init__(name)
        self.memory_type = memory_type
        self.capacity = capacity
        if memory_type == 'lru':
            self.states_value = LRUCache(self.capacity)
        if memory_type == 'random_dict':
            self.states_value = {}
        self.hit_counter = 0  # To track memory hits
        self.query_counter = 0

    def getHash(self, board):
        return str(board)

    def addState(self, state):
        state = str(state)
        self.states.append(state)

    def forget(self):
        if self.memory_type == 'lru':
            self.states_value = LRUCache(self.lru_capacity)
        if self.memory_type == 'random_dict':
            self.states_value = {}
        self.hit_counter = 0  # To track memory hits
        self.query_counter = 0

    def chooseAction(self, positions, current_board, current_trace, symbol, step):

        # print('board', current_board)

        action1, action2 = None, None
        # take random action
        while action1 == action2:
            idx1 = np.random.choice(len(positions))
            action1 = positions[idx1]
            idx2 = np.random.choice(len(positions))
            action2 = positions[idx2]

        if np.random.uniform(0, 1) > self.exp_rate:
            value_max = -999
            for p1 in positions:
                for p2 in positions:
                    if p1 != p2:
                        next_board = copy.deepcopy(current_board)
                        next_trace = copy.deepcopy(current_trace)
                        next_board[p1].append((step, symbol))
                        next_board[p2].append((step, symbol))
                        next_trace[p1].append((step, symbol))
                        next_trace[p2].append((step, symbol))
                        next_boardHash = self.getHash(next_board)
                        self.query_counter += 1
                        if self.states_value.get(next_boardHash) is None:
                            value = (random.random() - 0.5) / 10000
                        else:
                            value = self.states_value.get(next_boardHash)
                            self.hit_counter += 1
                        if value >= value_max:
                            value_max = value
                            action1 = p1
                            action2 = p2

        # print("{} takes action {}".format(self.name, action))
        # action1, action2 = 0, 2
        return action1, action2

    # Player does not know how to choose collapse yet
    def chooseCollapse(self, play, pos1, pos2, current_board=None, current_trace=None):
        choice = np.random.randint(2)
        if choice == 0:
            return pos1
        if choice == 1:
            return pos2

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        # Do not update value if it's in eval mode
        if self.is_eval:
            return
        if self.update_method == 'sarsa':
            for st in reversed(self.states):

                # Change for LRU structure
                if self.memory_type == 'lru':
                    if self.states_value.get(st) is None:
                        self.states_value.put(st, 0)
                    new_value = self.states_value.get(st) + self.lr * (self.decay_gamma * reward - self.states_value.get(st))
                    self.states_value.put(st, new_value)
                    reward = self.states_value.get(st)

                elif self.memory_type == 'random_dict':
                    if self.states_value.get(st) is None:
                        if len(self.states_value) >= self.capacity:
                            self.states_value.pop(random.choice(list(self.states_value.keys())))
                        self.states_value[st] = 0
                    self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
                    reward = self.states_value[st]

                else:
                    if self.states_value.get(st) is None:
                        self.states_value[st] = 0
                    self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
                    reward = self.states_value[st]