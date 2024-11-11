from pathlib import Path
from QAgent_new import Q_Agent
from Q_environment import Q_Environment
import numpy as np
import pickle
from policy_funcs import episilon_greedy, soft_max
from reward_funcs import reward_gas_level, reward_trace_area
from tqdm import tqdm
from AUV_plot_utils import plot_agent_behavior
import multiprocessing as mp
import logging

class Q_trainer:
    def __init__(self, env: Q_Environment, q_table_shape=(3, 3, 3, 3)) -> None:
        self._env: Q_Environment = env
        self._q_table = np.zeros(q_table_shape, dtype=np.int32)

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

def run_single_experiment(env_path, reward_func, policy_func, size, episodes, max_steps_per_episode, depth, store_q_table_by_episode = True) -> None:
    env = Q_Environment(env_path, depth=depth, x_bounds=(0, 250), y_bounds=(0, 250))
    trainer = Q_trainer(env)
    
    policy_name = policy_func.__name__
    reward_name = reward_func.__name__
    with tqdm(total=episodes, ncols=100, desc=f'Training agent {reward_name} {policy_name} (Depth {depth}, Size {size})', 
              bar_format='{l_bar}{bar} [elapsed: {elapsed} remaining: {remaining}]', colour='green', position=depth % 10) as pbar:
        for episode in range(episodes):
            agent = Q_Agent(trainer._env, reward_func=reward_func, policy=policy_func)
            agent.q_table = trainer._q_table
            agent.run(lawnmower_size=size, max_steps=max_steps_per_episode)
            trainer._q_table = agent.q_table
            if store_q_table_by_episode:
                filename = Path("./results") / "q_tables" / "q_tables_by_episodes" / policy_name / f"episode_{episode}_{reward_name}_{policy_name}_lawn_size_{size}.pkl"
                trainer.save_q_table(filename=filename)
            else:
                print(f"Episode {episode + 1}/{episodes} completed.")
            pbar.update(1)

        trainer._save_position_history(agent)
        if not store_q_table_by_episode:
            filename = Path("./results") / "q_tables" / policy_name / f"{episodes}_{reward_name}_lawn_size_{size}.pkl"
            trainer.save_q_table(filename=filename)

def run_experiments_for_depth(depth, episodes, max_steps_per_episode):
    env_path = Path(r"./sim/SMART-AUVs_OF-June-1c-0002.nc")
    all_tasks = []

    for reward_func in [reward_trace_area, reward_gas_level]:
        for policy_func in [episilon_greedy, soft_max]:
            for size in reversed((10, 20, 30, 40, 50, 60, 70, 80, 90, 100)):
                task = (env_path, reward_func, policy_func, size, episodes, max_steps_per_episode, depth)
                all_tasks.append(task)
    
    pool_size = max(1, mp.cpu_count() // len(range(64, 70)))
    with mp.Pool(processes=pool_size) as pool:
        pool.starmap(run_single_experiment, all_tasks)

def run_experiments() -> None:
    episodes = 50
    max_steps_per_episode = 2000
    depths = range(64, 70)
    
    processes = []
    
    for depth in depths:
        p = mp.Process(target=run_experiments_for_depth, args=(depth, episodes, max_steps_per_episode))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn') 
    run_experiments()

