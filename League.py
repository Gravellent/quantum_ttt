import numpy as np
import torch
import re
import copy
import matplotlib.pyplot as plt

from Player import *
from State import *


class League:
    '''
    Class that runs a league-like series of games
        players: list of Player objects (to start, should be even)
        state: board state class (NOTE: not an instantiated object)
        learning_games_per_round: number of games per round to learn from
        games_per_round: int that controls how many games a match up will play
        keep: non zero integer that denotes how many players will be kept per round (if 0 then will default to keeping all)
        warm_up: how many games to warm up with for each player (to address cold start issues)
    '''
    def __init__(self, players, state=QState, learning_games_per_round=100, games_per_round=101, keep=0, warm_up=100):
        # list of players
        self.players = players

        # number of warm up games
        self.warm_up = warm_up

        # how many to keep per round if implement elimination rounds
        if keep:
            self.keep = keep
        else:
            self.keep = len(self.players)

        # states for match ups
        self.state_type = state
        self.states = self.generate_states()

        # round information tracking
        self.games_per_round = games_per_round
        self.learning_games_per_round = learning_games_per_round
        self.round = 0
        self.round_stats = {}

        # may not need
        # self.next_round_schedule = self.play_schedule()

    '''
    Creates a states list of all the player match ups. All players go 1st and go 2nd.
    '''
    def generate_states(self):
        states = []
        for i in range(len(self.players)):
            for j in range(i+1,len(self.players)):
                p1 = self.players[i]
                p2 = self.players[j]

                # states for both players starting first
                states.append(self.state_type(p1, p2))
                states.append(self.state_type(p2, p1))
        return states

    # For creating match up schedule variations... not necessary right now
    def play_schedule(self):
        # states of pairs
        # randomly create pairs for next_round_schedule
        # round robin??
        # tournament style??
        # can be p1 and p2
        # do we want them all to play each other before elimination?
        raise("Not Implemented yet")

    '''
    Play 1 round of all match ups. Learning games iteratively, then eval games.
        games_per_step: int for number of games per learning step (to address cold start issues)
        random_train: boolean if want to iterate through the states randomly while training

        return: current round stats dict
    '''
    def play_round(self, games_per_step=2, random_train=False):
        self.round += 1
        # Learning
        for i in range(self.learning_games_per_round//games_per_step):
            if random_train:
                indices = np.arange(0, len(self.states))
            for matchup_state in self.states:
                if random_train:
                    i = np.random.randint(0, len(indices))
                    random_matchup_state = self.states[indices[i]]
                    indices = np.concatenate((indices[:i], indices[i+1:]))
                    self.play_match_up(random_matchup_state, learning_games=games_per_step, eval=False)

                else:
                    self.play_match_up(matchup_state, learning_games=games_per_step, eval=False)
        
        matchup_list = []
        stats = {}
        for s in range(len(self.states)):
            matchup_state = self.states[s]
            p1_name = matchup_state.p1.name
            p2_name = matchup_state.p2.name
            matchup_list.append(p1_name + " vs " + p2_name)

            result_state = self.play_match_up(matchup_state, eval_games=self.games_per_round, eval=True)
            
            p1_wins = result_state.p1_wins
            p2_wins = result_state.p2_wins
            ties = result_state.tie
            stats[s] = [ties, p1_wins, p2_wins]
             
        self.round_stats[self.round] = {"matchups":matchup_list,"stats":stats}
        return self.round_stats[self.round]
    
    '''
    Play a specific match up
        p1: player object
        p2: player object
        learn: boolean of whether or not to play learning games

        return: state object
    '''
    def play_match_up(self, state, learning_games=None, eval_games=None, eval=False, exp_rate=0.3):
        # learning games
        if not eval:
            if not learning_games:
                learning_games = self.learning_games_per_round
            state.p1.is_eval = False
            state.p2.is_eval = False
            state.p1.exp_rate = exp_rate
            state.p2.exp_rate = exp_rate
            state.play(learning_games)
        # eval games
        elif eval:
            if not eval_games:
                eval_games = self.games_per_round
            state.p1.is_eval = True
            state.p2.is_eval = True
            state.p1.exp_rate = 0
            state.p2.exp_rate = 0
            state.reset_metrics()
            state.play(eval_games)
        else:
            raise("Unforeseen eval value {}. Please provide a boolean.".format(eval))

        return state

    def warmup(self, warm_up=None, dummy=RandomQPlayer('dummy'), state=QState):
        if not warm_up:
            warm_up = self.warm_up
        for player in self.players:
            # player as P1
            dummy_matchup_state = state(player, dummy)
            self.play_match_up(dummy_matchup_state, learning_games=warm_up, eval=False)
            # player as P2
            dummy_matchup_state = state(dummy, player)
            self.play_match_up(dummy_matchup_state, learning_games=warm_up, eval=False)
            

    '''
    Save state of one player to a new copy of that player
        player: player object
        name: string name of new player, if not provided then a default version number will be concatenated or increased if already has one

        return: new player
    '''
    def fork_player(self, player, name=None):
        # get player type
        player_type = type(player)

        if name:
            new_name = name
        else:
            # check for version number
            version = re.findall(r"\_v\d+\_", player.name)
            # update version number
            if version:
                version = version[-1]
                v_num = int(re.findall(r"\d+", version)[0]) + 1
                new_name = re.sub(r"\_v\d+\_", "_v" + str(v_num) + "_", player.name)
            else:
                new_name = player.name + "_v1_"

        # initialize new player
        new_player = player_type(new_name)

        # assign the same state of player to new_player, copy so not same mem-pointer
        if 'states_value' in player.__dict__:
            new_player.states_value = player.states_value.copy()
        if 'model' in player.__dict__:
            new_player.model = copy.deepcopy(player.model)

        return new_player



if __name__ == "__main__":
    '''
    For Running a small League
    '''
    player_list = [DeepQPlayer("Challenger"), DeepQPlayer("Contender"), DeepQPlayer("Underdog")]
    # number of learning and eval games
    learning_games_per_iteration = 10
    eval_games_per_iteration = 10
    # how many rounds are playerd
    n_rounds = 3
    # warm up
    warmup_games = 10
    warmup = False
    # randomize order of training
    random_train = False
    games_per_step = 2
    # to get effective win rate - wins and 0.5*ties
    effective_win_rate = True
    # print stats list before and after plot
    verbose = True

    league = League(player_list, learning_games_per_round=learning_games_per_iteration, games_per_round=eval_games_per_iteration, warm_up=warmup_games)
    
    # NOTE: can use this for warm up
    if warmup:
        league.warmup()

    n_players = len(player_list)

    # {name: [[round_1_win, ...]
    #         [round_1_lose, ...] 
    #         [round_1_tie, ...]], ...}
    plot_stats = {}

    # play rounds of games
    for i in range(n_rounds):
        round_results = league.play_round(random_train=random_train, games_per_step=games_per_step)
        print("Round {}".format(i))
        # print(round_results)
        for match_up in round_results['stats'].keys():
            names = round_results['matchups'][match_up].split(' vs ')
            p1_name = names[0]
            p2_name = names[1]

            # Player 1 stats
            wins = round_results['stats'][match_up][1]
            lose = round_results['stats'][match_up][2]
            tie = round_results['stats'][match_up][0]
            # if not there
            if p1_name not in plot_stats:
                plot_stats[p1_name] = [[wins],[lose],[tie]]
            # if there
            else:
                # if round stats already initialized
                if len(plot_stats[p1_name][0]) == i+1:
                    plot_stats[p1_name][0][i] += wins
                    plot_stats[p1_name][1][i] += lose
                    plot_stats[p1_name][2][i] += tie
                # if round not initialized start column
                else:
                    plot_stats[p1_name][0].append(wins)
                    plot_stats[p1_name][1].append(lose)
                    plot_stats[p1_name][2].append(tie)                    

            # Player 2 stats
            wins = round_results['stats'][match_up][2]
            lose = round_results['stats'][match_up][1]
            tie = round_results['stats'][match_up][0]
            # if not there
            if p2_name not in plot_stats:
                plot_stats[p2_name] = [[wins],[lose],[tie]]
            # if there
            else:
                # if round stats already initialized
                if len(plot_stats[p2_name][0]) == i+1:
                    plot_stats[p2_name][0][i] += wins
                    plot_stats[p2_name][1][i] += lose
                    plot_stats[p2_name][2][i] += tie
                # if round not initialized start column
                else:
                    plot_stats[p2_name][0].append(wins)
                    plot_stats[p2_name][1].append(lose)
                    plot_stats[p2_name][2].append(tie)
    
    if verbose:
        print("\n", plot_stats)
    
    for player in plot_stats.keys():
        games = np.arange(0, (n_rounds)*eval_games_per_iteration, eval_games_per_iteration)
        e_w_r = []

        for i in range(n_rounds):
            # for effective win rate plot
            if effective_win_rate:
                e_w_r.append((plot_stats[player][0][i] + plot_stats[player][2][i]*0.5)/(eval_games_per_iteration*((n_players-1)*2)))

            plot_stats[player][0][i] /= eval_games_per_iteration*((n_players-1)*2)
            plot_stats[player][1][i] /= eval_games_per_iteration*((n_players-1)*2)
            plot_stats[player][2][i] /= eval_games_per_iteration*((n_players-1)*2)

        p1_plot = plt.plot(games, plot_stats[player][0], label="Win rate")
        p2_plot = plt.plot(games, plot_stats[player][1], label="Lose rate")
        tie_plot = plt.plot(games, plot_stats[player][2], label="Tie rate")
        plt.ylim(-0.1,1.1)
        plt.title("{} Stats".format(player))
        plt.legend()
        plt.show()

        # for effective win rate plot
        if effective_win_rate:
            e_w_r_plot = plt.plot(games, e_w_r, label="Effective Win rate")
            plt.ylim(-0.1,1.1)
            plt.title("{} Effective Win Rate".format(player))
            plt.legend()
            plt.show()

    if verbose:
        print("\n", plot_stats)

    for _round in plot_stats.keys():
        print("Player: {}".format(_round))
        print(plot_stats[_round])

    '''
    Test the League class
    '''
    # player_list = [RandomQPlayer("Challenger_v1_"), DeepQPlayer("Contender")]
    # test_league = League(player_list, learning_games_per_round=10, games_per_round=5)

    # print("Test play_match_up()")
    # state = test_league.states[0]
    # st = test_league.play_match_up(state)
    # print("p1 wins:",st.p1_wins)
    # print("p2 wins:",st.p2_wins)
    # print("ties:", st.tie)

    # print("\nTest for_player()")
    # new_player = test_league.fork_player(player_list[1])
    # print(new_player.name)
    # print("States values the same (should be True):", new_player.states_value==player_list[1].states_value)
    # print("Identical model weights (should be True):",(next(player_list[1].model.parameters())==next(new_player.model.parameters())).all().item())
    # print("Referencing same object (should be False):", player_list[1].model is new_player.model)

    # player_list = [RandomQPlayer("Challenger_v1_"), DeepQPlayer("Contender"), DeepQPlayer("Champion"), DeepQPlayer("Underdog")]
    # test_league = League(player_list, learning_games_per_round=10, games_per_round=5, warm_up=10)

    # print("\nLeague States")
    # print(test_league.states)

    # print("\nWarm up players", end="")
    # test_league.warmup()
    # print(" - Complete")

    # print("\nCreate and Play Rounds")
    # print("Round 1 results:")
    # round_result = test_league.play_round(random_train=True)
    # print(round_result)
    # print("\nResults after 2 rounds:")
    # round_result = test_league.play_round()
    # print(test_league.round_stats)

    
