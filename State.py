import numpy as np
import random

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIDE = 3
BOARD_SIZE = BOARD_SIDE * BOARD_SIDE

class State:
    def __init__(self, p1, p2, lose_reward=0, win_reward=1, p1_tie_reward=0.1, p2_tie_reward=0.5):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1
        self.results = []
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.p1_tie_reward = p1_tie_reward
        self.p2_tie_reward = p2_tie_reward

        # Metric tracking
        self.p1_wins = 0
        self.p2_wins = 0
        self.tie = 0
        self.games = 0


    def reset_metrics(self):
        self.p1_wins = 0
        self.p2_wins = 0
        self.tie = 0
        self.games = 0

    # The default way to get hash of board
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    # only when game ends
    def giveReward(self):
        result = self.winner()
        self.games += 1
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(self.win_reward)
            self.p2.feedReward(self.lose_reward)
            self.p1_wins += 1
        elif result == -1:
            self.p1.feedReward(self.lose_reward)
            self.p2.feedReward(self.win_reward)
            self.p2_wins += 1
        else:
            self.p1.feedReward(self.p1_tie_reward)
            self.p2.feedReward(self.p2_tie_reward)
            self.tie += 1

    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

class SPState(State):

    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        # col
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(diag_sum1, diag_sum2)
        if diag_sum == 3:
            self.isEnd = True
            return 1
        if diag_sum == -3:
            self.isEnd = True
            return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0 and self.collapsable() == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def collapsable(self):
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0.5 or self.board[i, j] == -0.5:
                    return 1
        return 0

    def collapse(self):
        index1 = np.argwhere(self.board == 0.5)
        index2 = np.argwhere(self.board == -0.5)
        seed1 = random.sample(range(0, len(index1)), int((len(index1) / 2)))
        seed2 = random.sample(range(0, len(index2)), int((len(index2) / 2)))

        for i1, j1 in enumerate(seed1):
            self.board[index1[j1][0], index1[j1][1]] = 1
        for i2, j2 in enumerate(seed2):
            self.board[index2[j2][0], index2[j2][1]] = -1
        for m in range(BOARD_ROWS):
            for n in range(BOARD_COLS):
                if self.board[m, n] != 1 and self.board[m, n] != -1:
                    self.board[m, n] = 0

        return self.board

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        #         print('avail', positions)
        return positions

    def updateState(self, position1, position2):
        self.board[position1] += self.playerSymbol * 0.5
        self.board[position2] += self.playerSymbol * 0.5
        #         print(len(self.availablePositions()))
        if len(self.availablePositions()) == 0 and self.collapsable() == 1:
            self.board = self.collapse()
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # only when game ends
    def giveReward(self):
        result = self.winner()
        self.games += 1
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
            self.p1_wins += 1
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
            self.p2_wins += 1
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)
            self.tie += 1


    def play(self, rounds=100, verbose=False):
        for i in range(rounds):
            #             if i%1000 == 0:
            #                 print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action1, p1_action2 = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and upate board state
                self.updateState(p1_action1, p1_action2)
                if verbose:
                    print('P1', self.board)

                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end
                #                 print('P1', self.board)

                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.results.append(win)
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action1, p2_action2 = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action1, p2_action2)
                    if verbose:
                        print('P2', self.board)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # ended with p2 either win or draw
                        self.results.append(win)
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

class ClassicState(State):

    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        # col
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(diag_sum1, diag_sum2)
        if diag_sum == 3:
            self.isEnd = True
            return 1
        if diag_sum == -3:
            self.isEnd = True
            return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def getHash(self):
        board_str = ''
        for pos in self.board.reshape(BOARD_COLS * BOARD_ROWS):
            if pos == -1:
                board_str += '2'
            else:
                board_str += str(int(pos))
        self.boardHash = board_str
        return self.boardHash

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def play(self, rounds=100):
        for i in range(rounds):
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    #                     self.results.append(win)           TODO RE-ADD BACK IN IF WANT TO USE
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        #                         self.results.append(win)       TODO RE-ADD BACK IN IF WANT TO USE
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')


class QState(State):

    def __init__(self, p1, p2):
        super().__init__(p1, p2)
        self.board = [[] for i in range(BOARD_SIZE)]
        self.trace = [[] for i in range(BOARD_SIZE)]
        self.availablePositions = [i for i in range(BOARD_SIZE)]
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1
        self.step = 0

    def winner(self):
        # row
        for i in range(BOARD_SIDE):
            flag = True
            score = 0
            for idx in range(i * BOARD_SIDE, i * BOARD_SIDE + BOARD_SIDE):
                if isinstance(self.board[idx], list):
                    flag = False
                    break
                score += self.board[idx]
            # print('score', score)
            if flag == True:
                if score == 3:
                    self.isEnd = True
                    return 1
                if score == -3:
                    self.isEnd = True
                    return -1

        # col
        for i in range(BOARD_SIDE):
            flag = True
            score = 0
            for idx in range(i, i + (BOARD_SIDE - 1) * BOARD_SIDE + 1, BOARD_SIDE):
                if isinstance(self.board[idx], list):
                    flag = False
                    break
                score += self.board[idx]
            if flag == True:
                if score == 3:
                    self.isEnd = True
                    return 1
                if score == -3:
                    self.isEnd = True
                    return -1

        # diagonal 1
        flag = True
        score = 0
        for idx in range(0, BOARD_SIZE, BOARD_SIDE + 1):
            if isinstance(self.board[idx], list):
                flag = False
                break
            score += self.board[idx]
        if flag == True:
            if score == 3:
                self.isEnd = True
                return 1
            if score == -3:
                self.isEnd = True
                return -1

        # diagonal 2
        flag = True
        score = 0
        for idx in range(BOARD_SIDE - 1, BOARD_SIZE - 1, BOARD_SIDE - 1):
            if isinstance(self.board[idx], list):
                flag = False
                break
            score += self.board[idx]
        if flag == True:
            if score == 3:
                self.isEnd = True
                return 1
            if score == -3:
                self.isEnd = True
                return -1

                # tie
        # no available positions
        if len(self.availablePositions) <= 1:
            return 0
        # not end
        self.isEnd = False
        return None

    def getHash(self):
        self.boardHash = str(self.board)
        return self.boardHash

    # Update available positions based on current state of the board.
    # This will be called after each collapse happens.
    def updateAvailablePositions(self):
        self.availablePositions = []
        for i in range(BOARD_SIZE):
            if isinstance(self.board[i], list):
                self.availablePositions.append(i)

    # Add player's moves to the appropriate positions in board
    def updateState(self, step, pos1, pos2):
        self.board[pos1].append((step, self.playerSymbol))
        self.board[pos2].append((step, self.playerSymbol))
        self.trace[pos1].append((step, self.playerSymbol))
        self.trace[pos2].append((step, self.playerSymbol))
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # Recursively search for the next entangled move.
    # Return true when goal is found, which means a cyclic entaglement is found.
    # Else return false.
    def searchNextEntangled(self, current, goal, pos):
        if current == goal:
            return True
        sPos = self.findSuperposition(current, pos, self.board)
        if len(self.board[sPos]) == 1:
            return False
        for neighbor in [p for p in self.board[sPos] if p != current]:
            search = self.searchNextEntangled(neighbor, goal, sPos)
            if search == True:
                return True
        return False

    # Return True if a cyclic entanglement is found.
    def isCyclicEntangled(self, play, pos1, pos2):
        if len(self.board[pos1]) == 1 or len(self.board[pos2]) == 1:
            return False

        nextEntangled = False
        for neighbor in [p for p in self.board[pos1] if p != play]:
            nextEntangled = self.searchNextEntangled(neighbor, play, pos1)
            if nextEntangled:
                return True

        return False

    # Given a play, find the superposition of the play.
    def findSuperposition(self, play, pos, board):
        for p in [i for i in self.availablePositions if i != pos]:
            if play in board[p]:
                return p
        return None

    # Given the chosen play and pos, collapse based on its cyclic entanglement.
    # This function can only be called after isCyclicEntangled returns True for the play and pos.
    def collapse(self, play, pos):
        self.board[pos] = play[1]

        for neighbor in [p for p in self.trace[pos] if p != play]:
            self.collapseNextEntangled(neighbor, play, pos)

        self.updateAvailablePositions()

    # Recursively collapse the state given a starting point.
    # This function can only be called on a starting point involved in cyclic entanglement.
    def collapseNextEntangled(self, current, goal, pos):
        if current == goal:
            return
        sPos = self.findSuperposition(current, pos, self.trace)
        self.board[sPos] = current[1]
        if len(self.trace[pos]) == 1:
            return
        for neighbor in [p for p in self.trace[sPos] if p != current]:
            if neighbor == goal:
                return
            self.collapseNextEntangled(neighbor, goal, sPos)

    # board reset
    def reset(self):
        super().reset()
        self.board = [[] for i in range(BOARD_SIZE)]
        self.trace = [[] for i in range(BOARD_SIZE)]
        self.availablePositions = [i for i in range(BOARD_SIZE)]
        self.isEnd = False
        self.playerSymbol = 1
        self.move = 0
        self.step = 0


    # Training
    def play(self, rounds=100):

        for i in range(rounds):

            cyclicEntangled = False
            while not self.isEnd:

                # Player 1
                # check cyclicEntangled
                # if True, choose collapse and update board state
                if cyclicEntangled:
                    collapse_pos = self.p1.chooseCollapse(play, pos1, pos2)
                    # print('collapsing...')
                    self.collapse(play, collapse_pos)
                    # print('end collapsing')
                    # self.showBoard()

                # check winner
                win = self.winner()
                if win is not None:
                    # if win == 1:
                    #     print(self.p1.name, "wins!")
                    # elif win == -1:
                    #     print(self.p2.name, "wins!")
                    # else:
                    #     print("tie!")
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                # get player action
                self.step += 1
                # print(self.board)
                # print(self.trace)
                positions = self.availablePositions
                pos1, pos2 = self.p1.chooseAction(positions, current_board=self.board, current_trace=self.trace,
                                                  symbol=self.playerSymbol, step=self.step)
                play = (self.step, 1)
                # take action and upate board state
                self.updateState(self.step, pos1, pos2)
                # self.showBoard()
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                cyclicEntangled = self.isCyclicEntangled(play, pos1, pos2)

                # Player 2
                # check cyclicEntangled
                # if True, choose collapse and update board state
                if cyclicEntangled:
                    collapse_pos = self.p2.chooseCollapse(play, pos1, pos2)
                    # print('collapsing...')
                    self.collapse(play, collapse_pos)
                    # print('end collapsing...')
                    # self.showBoard()

                # check winner
                win = self.winner()
                if win is not None:
                    # if win == 1:
                    #     print(self.p1.name, "wins!")
                    # elif win == -1:
                    #     print(self.p2.name, "wins!")
                    # else:
                    #     print("tie!")
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                # get player action
                self.step += 1
                positions = self.availablePositions
                pos1, pos2 = self.p2.chooseAction(positions, current_board=self.board, current_trace=self.trace,
                                                  symbol=self.playerSymbol, step=self.step)
                play = (self.step, -1)
                # take action and upate board state
                self.updateState(self.step, pos1, pos2)
                # self.showBoard()
                board_hash = self.getHash()
                self.p2.addState(board_hash)
                cyclicEntangled = self.isCyclicEntangled(play, pos1, pos2)


    # Play with human
    def play2(self):
        cyclicEntangled = False
        while not self.isEnd:

            # Player 1
            # check cyclicEntangled
            # if True, choose collapse and update board state
            if cyclicEntangled:
                collapse_pos = self.p1.chooseCollapse(play, pos1, pos2)
                print('collapsing...')
                self.collapse(play, collapse_pos)
                print('end collapsing')
                self.showBoard()

            # check winner
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                elif win == -1:
                    print(self.p2.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            # get player action
            self.step += 1
            # print(self.board)
            # print(self.trace)
            positions = self.availablePositions
            pos1, pos2 = self.p1.chooseAction(positions)
            play = (self.step, 1)
            # take action and upate board state
            self.updateState(self.step, pos1, pos2)
            self.showBoard()
            cyclicEntangled = self.isCyclicEntangled(play, pos1, pos2)

            # Player 2
            # check cyclicEntangled
            # if True, choose collapse and update board state
            if cyclicEntangled:
                collapse_pos = self.p2.chooseCollapse(play, pos1, pos2)
                print('collapsing...')
                self.collapse(play, collapse_pos)
                print('end collapsing...')
                self.showBoard()

            # check winner
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                elif win == -1:
                    print(self.p2.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            # get player action
            self.step += 1
            positions = self.availablePositions
            pos1, pos2 = self.p2.chooseAction(positions)
            play = (self.step, -1)
            # take action and upate board state
            self.updateState(self.step, pos1, pos2)
            self.showBoard()
            cyclicEntangled = self.isCyclicEntangled(play, pos1, pos2)

    # display current board
    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, BOARD_SIDE):
            print('-' * 34)
            out1 = '| '
            out2 = '| '
            out3 = '| '
            for j in range(0, BOARD_SIDE):
                moves = self.board[i * 3 + j]

                if isinstance(moves, int):
                    if moves == 1:
                        out1 = out1 + '\033[1m' + 'X' + '\033[0m' + ' ' * 8 + '| '
                    else:
                        out1 = out1 + '\033[1m' + 'O' + '\033[0m' + ' ' * 8 + '| '
                    out2 = out2 + ' ' * 9 + '| '
                    out3 = out3 + ' ' * 9 + '| '
                else:
                    if len(moves) == 0:
                        out1 = out1 + ' ' * 9 + '| '
                        out2 = out2 + ' ' * 9 + '| '
                        out3 = out3 + ' ' * 9 + '| '
                    else:
                        out1_temp = ''
                        out2_temp = ''
                        out3_temp = ''
                        for n in range(len(moves)):
                            if n < 3:
                                if moves[n][1] == 1:
                                    out1_temp = out1_temp + 'x' + str(moves[n][0]) + ' '
                                else:
                                    out1_temp = out1_temp + 'o' + str(moves[n][0]) + ' '
                            elif n < 6:
                                if moves[n][1] == 1:
                                    out2_temp = out2_temp + 'x' + str(moves[n][0]) + ' '
                                else:
                                    out2_temp = out2_temp + 'o' + str(moves[n][0]) + ' '
                            else:
                                if moves[n][1] == 1:
                                    out3_temp = out3_temp + 'x' + str(moves[n][0]) + ' '
                                else:
                                    out3_temp = out3_temp + 'o' + str(moves[n][0]) + ' '
                        out1 = out1 + out1_temp + ' ' * (9 - len(out1_temp)) + '| '
                        out2 = out2 + out2_temp + ' ' * (9 - len(out2_temp)) + '| '
                        out3 = out3 + out3_temp + ' ' * (9 - len(out3_temp)) + '| '
            print(out1)
            print(out2)
            print(out3)
        print('-' * 34)

