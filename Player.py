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


class ClassicPlayer(Player):
    def __init__(self, name, exp_rate=0.3, player_symbol=1, update_method='sarsa'):
        super().__init__(name, exp_rate=exp_rate,
                         player_symbol=player_symbol, update_method=update_method)

    def getHash(self, board):
        board_str = ''
        for pos in board.reshape(BOARD_COLS * BOARD_ROWS):
            if pos == -1:
                board_str += '2'
            else:
                board_str += str(int(pos))
        return board_str

    # Find the distance between states
    def is_next_state(self, old_state, new_state):
        diff = []
        for pos1, pos2 in zip(old_state, new_state):
            if pos1 == pos2:
                continue
            if pos1 == '0' and pos2 != '0':
                diff.append(pos2)
            else:
                return False
        return sorted(diff) == ['1','2']

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        if self.is_eval:
            return
        if self.update_method == 'sarsa':
            for st in reversed(self.states):
                if self.states_value.get(st) is None:
                    self.states_value[st] = 0
                self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
                reward = self.states_value[st]

        if self.update_method == 'expected_sarsa':
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


class RandomClassicPlayer(Player):
    def __init__(self, name):
        self.name = name
        self.update_method = "Random Player"

    def chooseAction(self, positions, current_board, symbol):
        idx = np.random.choice(len(positions))
        action = positions[idx]
        return action

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass


class SPPlayer(Player):

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

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        # Do not update value if it's in eval mode
        if self.is_eval:
            return
        if self.update_method == 'sarsa':
            for st in reversed(self.states):
                if self.states_value.get(st) is None:
                    self.states_value[st] = 0
                self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
                reward = self.states_value[st]

        # if self.update_method == 'expected_sarsa':
        #     #             print('ss', self.states, 'reward', reward)
        #     for st in reversed(self.states):
        #         if self.states_value.get(st) is None:
        #             self.states_value[st] = 0
        #         possible_states = []
        #         if reward is None:
        #             for next_st in self.states_value:
        #                 if self.is_next_state(st, next_st):
        #                     possible_states.append((next_st, self.states_value[next_st]))
        #             reward = max([_[1] for _ in possible_states])
        #         #                     print(possible_states, reward)
        #         self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
        #         #                 print(self.states_value)
        #         reward = None


class RandomSPPlayer(Player):
    def __init__(self, name):
        self.name = name
        self.update_method = "Random Player"

    def chooseAction(self, positions, current_board, symbol):
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


class DeepQPlayer(Player):

    def __init__(self, name, model_cls=LinearModel):
        super().__init__(name)
        self.model = model_cls()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def forget(self):
        self.model = LinearModel()

    def getHash(self, board):
        return str(board)

    def addState(self, state):
        self.states.append(state)

    def get_state_tensor(self, board):
        state_tensor = torch.zeros(9, 9)
        turn_ent_dict = defaultdict(list)
        for pos, val in enumerate(board):
            # If it's a list or tensor, and not empty, means there's an entanglement
            if type(val) == list:
                for turn, symbol in val:
                    turn_ent_dict[turn].append((pos, symbol))
            # Collapsed cell takes a diagonal position in tensor
            if type(val) == int:
                state_tensor[pos, pos] = val

        # Add the entanglement to the state tensor
        for v in turn_ent_dict.values():
            assert(len(v) == 2)
            pos1, symbol1 = v[0]
            pos2, symbol2 = v[1]
            assert(symbol1 == symbol2)
            state_tensor[pos1, pos2] = symbol1
            state_tensor[pos2, pos1] = symbol1
        return state_tensor

    def get_value(self, board):
        state_tensor = self.get_state_tensor(board)
        value = self.model(state_tensor)
        return value.item()

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
                        value = self.get_value(next_board)
                        if value >= value_max:
                            value_max = value
                            action1 = p1
                            action2 = p2

        # print("{} takes action {}".format(self.name, action1, action2))
        return action1, action2

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        # Do not update value if it's in eval mode
        if self.is_eval:
            return
        # if self.update_method == 'sarsa':
        for st in reversed(self.states):
            state_tensor = self.get_state_tensor(st)
            y_pred = self.model(state_tensor)
            y_true = torch.FloatTensor([reward])
            loss = self.criterion(y_pred, y_true)
            loss.backward()
            self.optimizer.step()
            reward = self.decay_gamma * reward

    def collapse(self, play, pos, board, trace):
        board[pos] = play[1]
        for neighbor in [p for p in trace[pos] if p != play]:
            board = self.collapseNextEntangled(neighbor, play, pos, board, trace)
        return board

    def collapseNextEntangled(self, current, goal, pos, board, trace):
        if current == goal:
            return board
        sPos = self.findSuperposition(current, pos, board, trace)
        board[sPos] = current[1]
        
        if len(trace[pos]) == 1:
            return board
        for neighbor in [p for p in trace[sPos] if p != current]:
            if neighbor == goal:
                return board
            return self.collapseNextEntangled(neighbor, goal, sPos, board, trace)
        return board

    def findSuperposition(self, play, pos, board, trace):
        availablePositions = self.availablePositions(board)
        for p in [i for i in availablePositions if i != pos]:
            if play in board[p]:
                return p
        return None

    def availablePositions(self, board):
        positions = []
        for i in range(BOARD_ROWS*BOARD_COLS):
            if isinstance(board[i], list):
                positions.append(i)
        return positions

    # Player does not know how to choose collapse yet
    def chooseCollapse(self, play, pos1, pos2, current_board, current_trace):

        # Random if exploring
        choice = np.random.randint(2)
        if choice == 0:
            action = pos1
        if choice == 1:
            action = pos2

        if np.random.uniform(0, 1) > self.exp_rate:
            value_max = -999
            for pos in (pos1, pos2):
                next_board = copy.deepcopy(current_board)
                next_trace = copy.deepcopy(current_trace)
                print('board before collapse:',next_board) 
                next_board = self.collapse(play, pos, next_board, next_trace)
                print('next board:', next_board)
                value = self.get_value(next_board)
                if value >= value_max:
                    value_max = value
                    action = pos
#                 next_board[pos] = play[1]
#                 for neighbor in [p for p in next_trace[pos] if p != play]:
#                     print("current board", next_board)
#                     next_board = self.collapse(play, pos, next_board, avail_pos, next_trace)
#                     print("next board:", next_board)
#                     value = self.get_value(next_board)
#                     if value >= value_max:
#                         value_max = value
#                         action = pos
        return action


class QPlayer(Player):

    def getHash(self, board):
        return str(board)

    def addState(self, state):
        self.states.append(state)

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
                        value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(
                            next_boardHash)
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
                if self.states_value.get(st) is None:
                    self.states_value[st] = 0
                self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
                reward = self.states_value[st]


class RandomQPlayer(Player):

    def __init__(self, name):
        self.name = name
        self.update_method = "Random Player"

    def chooseAction(self, positions,  current_board, current_trace, symbol, step):
        pos1 = np.random.randint(len(positions))
        pos2 = np.random.randint(len(positions))
        while pos2 == pos1:
            pos2 = np.random.randint(len(positions))
        # print(positions[pos1], positions[pos2])
        return (positions[pos1], positions[pos2])


    def chooseCollapse(self, play, pos1, pos2, current_board=None, current_trace=None):
        choice = np.random.randint(2)
        if choice == 0:
            return pos1
        if choice == 1:
            return pos2


class HumanQPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions,  current_board, current_trace, symbol, step):
        while True:
            row1 = int(input("Input your first action row:"))
            col1 = int(input("Input your fisrt action col:"))
            row2 = int(input("Input your second action row:"))
            col2 = int(input("Input your second action col:"))
            pos1 = row1 * BOARD_SIDE + col1
            pos2 = row2 * BOARD_SIDE + col2
            if pos1 in positions and pos2 in positions:
                return (pos1, pos2)


    def chooseCollapse(self, play, pos1, pos2):
        print('!! A cyclic entanglement occurs, below are 2 possible collapses:')
        if play[1] == 1:
            print('Play: ' + 'x' + str(play[0]))
        else:
            print('Play: ' + 'o' + str(play[0]))
        row1 = pos1 // BOARD_SIDE
        row2 = pos2 // BOARD_SIDE
        col1 = pos1 % BOARD_SIDE
        col2 = pos2 % BOARD_SIDE
        print('choice 1:', 'row', row1, 'col', col1)
        print('choice 2:', 'row', row2, 'col', col2)
        while True:
            choice = int(input("Input your choice for collapse:"))
            if choice == 1:
                return row1 * BOARD_SIDE + col1
            if choice == 2:
                return row2 * BOARD_SIDE + col2
