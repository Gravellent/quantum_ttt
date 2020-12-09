from Player.BasePlayer import *

class SPPlayer(BasePlayer):

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