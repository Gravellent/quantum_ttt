from Player.BasePlayer import *


class ClassicPlayer(BasePlayer):
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