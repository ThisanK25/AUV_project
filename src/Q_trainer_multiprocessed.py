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

    def train(self, episodes=50, max_steps_per_episode=500, lawnmover_size=70, reward_func = reward_gas_level, policy = episilon_greedy, store_q_table_by_episode = False) -> None:
        policy_name:str = policy.__name__
        reward_name:str = reward_func.__name__
        with tqdm(total=episodes, ncols=100, desc=f'Training agent {reward_name} {policy_name}', bar_format='{l_bar}{bar} [elapsed: {elapsed} remaining: {remaining}]'
, colour='green', position=0) as pbar:
            for episode in range(episodes):
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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Make sure to define or import all necessary functions and classes

def run_single_experiment(env_path, reward_func, policy_func, size, episodes, max_steps_per_episode, depth):
    logging.debug("Starting run_single_experiment for depth %d, size %d", depth, size)
    env = Q_Environment(env_path, depth=depth, x_bounds=(0, 250), y_bounds=(0, 250))
    trainer = Q_trainer(env)
    
    trainer.train(episodes=episodes, max_steps_per_episode=max_steps_per_episode, 
                  lawnmover_size=size, reward_func=reward_func, policy=policy_func, 
                  store_q_table_by_episode=True)
    
    reward_func_name = reward_func.__name__
    policy_func_name = policy_func.__name__
    figure_name = f"./results/plots/{policy_func_name}_{reward_func_name}/" \
                  f"training_lawnmover_size_{size}_steps_per_episode_{max_steps_per_episode}_" \
                  f"reward_trace_area_depth_{depth}.png"
    
    plot_agent_behavior(
        chemical_file_path=str(env_path),
        time_target=0,
        z_target=depth,
        data_parameter='pH',
        figure_name=figure_name,
        position_history=trainer.position_history
    )
    logging.debug("Completed run_single_experiment for depth %d, size %d", depth, size)

def run_experiments_for_depth(depth, episodes, max_steps_per_episode):
    logging.debug("Starting experiments for depth %d", depth)
    env_path = Path(r"./sim/SMART-AUVs_OF-June-1c-0002.nc")
    all_tasks = []

    for reward_func in [reward_trace_area, reward_gas_level]:
        for policy_func in [episilon_greedy, soft_max]:
            for size in reversed((10, 20, 30, 40, 50, 60, 70, 80, 90, 100)):
                task = (env_path, reward_func, policy_func, size, episodes, max_steps_per_episode, depth)
                all_tasks.append(task)
    
    pool_size = max(1, mp.cpu_count() // len(range(64, 70)))
    logging.debug("Pool size for depth %d: %d", depth, pool_size)
    with mp.Pool(processes=pool_size) as pool:
        pool.starmap(run_single_experiment, all_tasks)
    logging.debug("Completed experiments for depth %d", depth)

def run_experiments() -> None:
    episodes = 50
    max_steps_per_episode = 2000
    depths = range(64, 70)
    
    processes = []
    
    for depth in depths:
        p = mp.Process(target=run_experiments_for_depth, args=(depth, episodes, max_steps_per_episode))
        p.start()
        logging.debug("Started process for depth %d", depth)
        processes.append(p)
    
    for p in processes:
        p.join()
        logging.debug("Joined process %s" % str(p))

if __name__ == "__main__":
    mp.set_start_method('spawn')  
    run_experiments()