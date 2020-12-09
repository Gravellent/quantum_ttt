from Player.BasePlayer import *

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