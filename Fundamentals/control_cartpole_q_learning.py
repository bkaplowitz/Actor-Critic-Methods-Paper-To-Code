import numpy as np

class Agent():
    def __init__(self, lr, gamma, n_actions, state_space, eps_start, eps_end,
                 eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.actions = list(range(n_actions))
        self.states = state_space
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.Q = {}

        self.init_Q()

    def init_Q(self):
        for state in self.states:
            for action in self.actions:
                self.Q[(state, action)] = 0.0

    def max_action(self, state):
        actions = np.array([self.Q[(state, a)] for a in self.actions])
        return np.argmax(actions)

    def choose_action(self, state):
        return (
            np.random.choice(self.actions)
            if np.random.random() < self.epsilon
            else self.max_action(state)
        )

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                if self.epsilon>self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        a_max = self.max_action(state_)

        self.Q[(state, action)] = self.Q[(state, action)] + self.lr*(reward +
                                        self.gamma*self.Q[(state_, a_max)] -
                                        self.Q[(state, action)])

