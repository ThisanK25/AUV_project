from pathlib import Path
from QAgent_new import Q_Agent
from Q_environment import Q_Environment
import numpy as np
import pickle
from policy_funcs import episilon_greedy, soft_max
from reward_funcs import reward_gas_level, reward_trace_area
from tqdm import tqdm
from AUV_plot_utils import plot_agent_behavior

class Q_trainer:
    def __init__(self, env: Q_Environment, q_table_shape=(3, 3, 3, 3)) -> None:
        self._env: Q_Environment = env
        self._q_table = np.zeros(q_table_shape, dtype=np.int32)

    def train(self, episodes=50, max_steps_per_episode=2000, lawnmover_size=70, reward_func = reward_gas_level, policy = episilon_greedy, store_q_table_by_episode = False) -> None:
        policy_name:str = policy.__name__
        reward_name:str = reward_func.__name__

        for episode in range(episodes):
            agent = Q_Agent(self._env, reward_func=reward_func, policy=policy)
            agent.q_table = self._q_table
            agent.run(lawnmower_size=lawnmover_size, max_steps=max_steps_per_episode)
            self._q_table = agent.q_table
            if store_q_table_by_episode:
                filename = Path("./results") / "q_tables" / "q_tables_by_episodes" / policy_name / f"episode_{episode}_{reward_name}_depth_{self._env.depth}_lawn_size_{lawnmover_size}.pkl"
                self.save_q_table(filename=filename)
            else:
                print(f"Episode {episode + 1}/{episodes} completed.")

        self._save_position_history(agent)
        if not store_q_table_by_episode:
            filename = Path("./results") / "q_tables" / policy_name / f"{episode}_depth_{self._env.depth}_lawn_size_{lawnmover_size}.pkl"
            self.save_q_table(filename=filename)

    def _save_position_history(self, agent: Q_Agent) -> None:
        self._position_history = agent.position_history
    
    def save_q_table(self, filename:Path = Path(r"./results/q_tables/q_table.pkl")) -> None:
        # Eunsure that the path exists
        filename.parent.mkdir(exist_ok=True, parents=True)
        with open(filename, "wb") as f:
            pickle.dump(self._q_table, f)
    
    @property
    def position_history(self) -> list:
        if hasattr(self, "_position_history"):
            return self._position_history
        return []

def run_experiments() -> None:
    episodes =   50
    max_steps_per_episode = 2000
    # 10 lawn sizes, 2 depths, 2 reward functions, 2 policy functions.
    total_training_runs = 10 * 2 * 2 * 2
    with tqdm(total=total_training_runs, ncols=100, desc=f'Training runs completed', bar_format='{l_bar}{bar} \033[94m [elapsed: {elapsed} remaining: {remaining}]'
                , colour='green', position=0) as pbar:
        for depth in (64, 67):
            env = Q_Environment(Path(r"./sim/SMART-AUVs_OF-June-1c-0002.nc"), depth=depth, x_bounds=(0, 250), y_bounds=(0, 250))
            for reward_func in [reward_trace_area, reward_gas_level]:
                for policy_func in [episilon_greedy, soft_max]:
                    for size in reversed((10, 20, 30, 40, 50, 60, 70, 80, 90, 100)):
                        trainer = Q_trainer(env)
                        trainer.train(episodes=episodes, max_steps_per_episode=max_steps_per_episode, lawnmover_size=size, reward_func = reward_func, policy=policy_func, store_q_table_by_episode=True)
                        pbar.update(1)

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
