from Player.BasePlayer import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DeepQPlayer(BasePlayer):

    def __init__(self, name, model_cls=LinearModel):
        super().__init__(name)
        self.model_cls = model_cls
        self.model = model_cls().to(DEVICE)
        self.criterion = nn.MSELoss().to(DEVICE)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def forget(self):
        self.model = self.model_cls().to(DEVICE)

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
            assert (len(v) == 2)
            pos1, symbol1 = v[0]
            pos2, symbol2 = v[1]
            assert (symbol1 == symbol2)
            state_tensor[pos1, pos2] = symbol1
            state_tensor[pos2, pos1] = symbol1
        return state_tensor * self.player_symbol

    def get_value(self, board):
        state_tensor = self.get_state_tensor(board).to(DEVICE)
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
                        # print(next_board, value, value_max, p1, p2)
                        if value >= value_max:
                            value_max = value
                            action1 = p1
                            action2 = p2

        # print("{} takes action {}".format(self.name, action1, action2))
        # print(action1, action2)
        return action1, action2

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        # Do not update value if it's in eval mode
        if self.is_eval:
            return
        # if self.update_method == 'sarsa':
        for st in reversed(self.states):
            self.optimizer.zero_grad()
            state_tensor = self.get_state_tensor(st).to(DEVICE)
            y_pred = self.model(state_tensor)
            y_true = torch.FloatTensor([reward]).to(DEVICE)
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
            board = self.collapseNextEntangled(neighbor, goal, sPos, board, trace)
        return board

    def findSuperposition(self, play, pos, board, trace):
        availablePositions = self.availablePositions(board)
        for p in [i for i in availablePositions if i != pos]:
            if play in board[p]:
                return p
        return None

    def availablePositions(self, board):
        positions = []
        for i in range(BOARD_ROWS * BOARD_COLS):
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
                next_board = self.collapse(play, pos, next_board, next_trace)
                value = self.get_value(next_board)
                if value >= value_max:
                    value_max = value
                    action = pos
        return action