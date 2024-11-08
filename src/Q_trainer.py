from pathlib import Path
from QAgent_new import Q_Agent
from Q_environment import Q_Environment
import numpy as np
import pickle
from policy_funcs import episilon_greedy, soft_max
from reward_funcs import reward_gas_level, reward_trace_area
from tqdm import tqdm

class Q_trainer:
    def __init__(self, env: Q_Environment, q_table_shape=(3, 3, 3, 3)) -> None:
        self._env: Q_Environment = env
        self._q_table = np.zeros(q_table_shape, dtype=np.int32)

    def train(self, episodes=50, max_steps_per_episode=500, lawnmover_size=70, reward_func = reward_gas_level, policy = episilon_greedy, store_q_table_by_episode = False) -> None:
        policy_name:str = policy.__name__
        reward_name:str = reward_func.__name__
        with tqdm(total=episodes, ncols=100, desc=f'Training agent {reward_name} {policy_name}', bar_format='{l_bar}{bar} [elapsed: {elapsed} remaining: {remaining}]'
, colour='green', position=0) as pbar:
            for episode in range(episodes+1):
                agent = Q_Agent(self._env, reward_func=reward_func, policy=policy)
                agent.q_table = self._q_table
                agent.run(lawnmower_size=lawnmover_size, max_steps=max_steps_per_episode)
                self._q_table = agent.q_table
                if store_q_table_by_episode:
                    filename = Path("./results") / "q_tables" / "q_tables_by_episodes" / policy_name / f"episode_{episode}_{reward_name}_{policy_name}_lawn_size_{lawnmover_size}.pkl"
                    self.save_q_table(filename=filename)
                else:
                    print(f"Episode {episode + 1}/{episodes} completed.")
                # if episode % 100 == 0:
                #     print(f"Actions performed in episode {episode}: {agent._actions_performed}")
                pbar.update()

            self._save_position_history(agent)
            if not store_q_table_by_episode:
                filename = Path("./results") / "q_tables" / policy_name / f"{episode}_{reward_name}_lawn_size_{lawnmover_size}.pkl"
                self.save_q_table(filename=filename)

    def _save_position_history(self, agent: Q_Agent) -> None:
        self.position_history = agent._actions_performed
    
    def save_q_table(self, filename:Path = Path(r"./results/q_tables/q_table.pkl")) -> None:
        # Eunsure that the path exists
        filename.parent.mkdir(exist_ok=True, parents=True)
        with open(filename, "wb") as f:
            pickle.dump(self._q_table, f)

def run_experiments() -> None:
    episodes =   50
    max_steps_per_episode = 5000
    for size in (50,):
        for depth in range(66, 67):
            env = Q_Environment(Path(r"./sim/SMART-AUVs_OF-June-1c-0002.nc"), depth=depth, x_bounds=(0, 250), y_bounds=(0, 250))
            trainer = Q_trainer(env)
            trainer.train(episodes=episodes, max_steps_per_episode=max_steps_per_episode, lawnmover_size=size, reward_func = reward_trace_area, policy=episilon_greedy, store_q_table_by_episode=True)
            # for zoom in [False]:
            #     figure_name = rf"./results/plots/softmax_reward_trace_area/training_lawnmover_size_{size}_steps_per_episode{max_steps_per_episode}_reward_trace_area_depth_{depth}.png"
            #     trainer.plot_behavior(
            #         chemical_file_path=r"./sim/SMART-AUVs_OF-June-1c-0002.nc",
            #         time_target=0,
            #         z_target=depth,
            #         data_parameter='pH',
            #         zoom=zoom,
            #         figure_name=figure_name
            #     )

if __name__ == "__main__":
    run_experiments()
    # env = Q_Environment(Path(r"./sim/SMART-AUVs_OF-June-1c-0002.nc"), depth=68, x_bounds=(0, 250), y_bounds=(0, 250))
    # trainer = Q_trainer(env)
    # trainer.train(episodes=10, max_steps_per_episode=5000, lawnmover_size=70)
    
    # # Example of plotting behavior (adjust the chemical_file_path, time_target, etc. as necessary)
    # trainer.plot_behavior(
    #     chemical_file_path=r"./sim/SMART-AUVs_OF-June-1c-0002.nc",  
    #     time_target=0,
    #     z_target=68,
    #     data_parameter='pH',
    #     zoom=False
    # )
