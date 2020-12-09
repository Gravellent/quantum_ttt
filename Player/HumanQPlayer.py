from Player.BasePlayer import *

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