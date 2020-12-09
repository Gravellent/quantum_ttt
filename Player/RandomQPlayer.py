from Player.BasePlayer import *

class RandomQPlayer(BasePlayer):

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