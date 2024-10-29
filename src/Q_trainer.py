from pprint import pprint
from QAgent_new import Q_Agent
from Q_environment import Q_Environment
import numpy as np


class Q_trainer:
    def __init__(self, env:Q_Environment, q_table_shape:tuple[int, int, int, int] = (3, 3, 3, 3)):
        self._env = env
        self._q_table = np.zeros_like(q_table_shape, dtype=np.int32)

    
    def train(self, episodes = 1000, max_steps_per_episode = 100):
        # lawnmover goes here
        for episode in range(episodes):
            agent = Q_Agent(self._env)
            agent.q_table = self._q_table
            agent.run()
            self._q_table = agent._q_table

            if episode % 10 == 0:
                pprint(self._q_table)
