import csv
import os
import random
from pathlib import Path

import numpy as np
from keras.layers import Dense
from keras.models import Sequential, load_model


class Agent:
    def __init__(self, first_move, exploration_factor=1):
        self.epsilon = 0.1
        self.alpha = 0.5
        self.prev_state = np.zeros((3, 3))
        self.state = None
        self.first_move = first_move
        self.exp_factor = exploration_factor

    def calc_value(self, state):
        pass

    def learn_state(self, state, winner):
        pass

    def make_move(self, state, winner):
        self.state = state

        if winner is not None:
            new_state = state
            return new_state

        p = random.uniform(0, 1)
        if p < self.exp_factor:
            # exploitation
            new_state = self.make_optimal_move(state)
        else:
            # exploration
            moves = [s for s, v in np.ndenumerate(state) if v == 0]
            r = random.choice(moves)
            new_state = np.copy(state)
            if self.first_move:
                new_state[r] = 1
            else:
                new_state[r] = 2

        return new_state

    def make_move_and_learn(self, state, winner):
        self.learn_state(state, winner)

        return self.make_move(state, winner)

    def make_optimal_move(self, state):
        moves = [s for s, v in np.ndenumerate(state) if v == 0]

        if len(moves) == 1:
            # there is only one move possible
            new_state = np.copy(state)
            if self.first_move:
                new_state[moves[0]] = 1
            else:
                new_state[moves[0]] = 2
            return new_state

        temp_state_list = []
        v = -float('Inf')

        for x in moves:
            v_temp = []
            temp_state = np.copy(state)
            if self.first_move:
                temp_state[x] = 1
            else:
                temp_state[x] = 2

            moves_op = [s for s, v in np.ndenumerate(state) if v == 0]
            for y in moves_op:
                temp_state_op = np.copy(temp_state)
                if not self.first_move:
                    temp_state_op[y] = 1
                else:
                    temp_state_op[y] = 2
                v_temp.append(self.calc_value(temp_state_op))

            # deletes Nones
            v_temp = list(filter(None.__ne__, v_temp))

            if len(v_temp) != 0:
                v_temp = np.min(v_temp)
            else:
                # encourage exploration
                v_temp = 1

            if v_temp > v:
                temp_state_list = [temp_state]
                v = v_temp
            elif v_temp == v:
                temp_state_list.append(temp_state)

        try:
            new_state = random.choice(temp_state_list)
        except ValueError:
            print('temp state:', temp_state_list)
            raise Exception('temp state empty')

        return new_state

    def reward(self, winner):
        if winner == "white":
            if self.first_move:
                r = 1
            else:
                r = -1
        elif winner == "black":
            if not self.first_move:
                r = 1
            else:
                r = -1
        elif winner is None:
            r = 0
        else:
            r = 0.5
        return r


class QAgent(Agent):
    def __init__(self, first_move, exploration_factor=1):
        super().__init__(first_move, exploration_factor)
        self.values = dict()
        self.load_values()

    def learn_state(self, state, winner):
        prev_state_str = np.array2string(self.prev_state.reshape(9).astype(int), separator='')
        state_str = np.array2string(state.reshape(9).astype(int), separator='')
        aux = '1' if self.first_move else '2'
        if aux in state_str:
            if prev_state_str in self.values.keys():
                v_s = self.values[prev_state_str]
            else:
                v_s = int(0)

            r = self.reward(winner)

            if state_str in self.values.keys() and winner is None:
                v_s_tag = self.values[state_str]
            else:
                v_s_tag = int(0)

            self.values[prev_state_str] = v_s + self.alpha*(r + v_s_tag - v_s)

        self.prev_state = state

    def calc_value(self, state):
        state_str = np.array2string(state.reshape(9).astype(int), separator='')
        if state_str in self.values.keys():
            return self.values[state_str]

    def load_values(self):
        aux = 'white' if self.first_move else 'black'
        s = 'datasets/qvalues_' + aux + '.csv'
        try:
            value_csv = csv.reader(open(s, 'r'))
            for row in value_csv:
                k, v = row
                self.values[k] = float(v)
        except:
            pass
        print("Loaded q_agent values.")

    def save_values(self):
        aux = 'white' if self.first_move else 'black'
        s = 'datasets/qvalues_' + aux + '.csv'
        try:
            os.remove(s)
        except:
            pass
        a = csv.writer(open(s, 'a', newline=''))

        for v, k in self.values.items():
            a.writerow([v, k])
        print("Saved q_agent values.")


class DeepAgent(Agent):
    def __init__(self, first_move, exploration_factor=1):
        super().__init__(first_move, exploration_factor)
        self.value_model = self.load_model()

    def learn_state(self, state, winner):
        target = self.calc_target(state, winner)
        self.train_model(target, 10)
        self.prev_state = state

    def load_model(self):
        aux = 'white' if self.first_move else 'black'
        s = 'datasets/model_values_' + aux + '.h5'
        model_file = Path(s)
        if model_file.is_file():
            model = load_model(s)
            print('load model: ' + s)
        else:
            print('new model')
            model = Sequential()
            model.add(Dense(18, activation='relu', input_shape=(9,)))
            model.add(Dense(18, activation='relu'))
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

        return model

    def calc_value(self, state):
        return self.value_model.predict(state.reshape(1, 9))

    def calc_target(self, state, winner):
        aux = 1 if self.first_move else 2
        if aux in state:
            v_s = self.calc_value(self.prev_state)
            r = self.reward(winner)

            if winner is None:
                v_s_tag = self.calc_value(state)
            else:
                v_s_tag = 0

            target = np.array(v_s + self.alpha * (r + v_s_tag - v_s))

            return target

    def train_model(self, target, epochs):
        if target is not None:
            self.value_model.fit(self.prev_state.reshape(1, 9), target, epochs=epochs, verbose=0)

    def save_values(self):
        aux = 'white' if self.first_move else 'black'
        s = 'datasets/model_values_' + aux + '.h5'
        try:
            os.remove(s)
        except:
            pass
        self.value_model.save(s)
