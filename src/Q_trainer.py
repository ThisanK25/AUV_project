from pathlib import Path
from pprint import pprint
from QAgent_new import Q_Agent
from Q_environment import Q_Environment
import numpy as np
import pickle # Too store the q-table



class Q_trainer:
    def __init__(self, env:Q_Environment, q_table_shape = (3, 3, 3, 3)):
        self._env = env
        self._q_table = np.zeros(q_table_shape, dtype=np.int32)

    
    def train(self, episodes = 500, max_steps_per_episode = 2000):
        for episode in range(episodes):
            agent = Q_Agent(self._env)
            agent.q_table = self._q_table
            agent.run(max_steps_per_episode)
            self._q_table = agent._q_table

            if episode % 100 == 0:
                print(agent._actions_performed)
        pprint(self._q_table)
        with open("q_table.pkl", "xb") as f:
            pickle.dump(self._q_table, f)


if __name__ == "__main__":
    env = Q_Environment(Path(r"./sim/SMART-AUVs_OF-June-1c-0002.nc"), depth=68)
    trainer = Q_trainer(env)
    trainer.train()
    
