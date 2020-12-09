from Player.BasePlayer import *

class RandomClassicPlayer(BasePlayer):
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