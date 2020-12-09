from Player.BasePlayer import *

class QPlayer(BasePlayer):

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