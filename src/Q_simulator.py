from pathlib import Path
import pickle
import reward_funcs
import policy_funcs
from tqdm import tqdm
from QAgent_Enums import PH_Reading
from QAgent_new import Q_Agent
from Q_environment import Q_Environment



class Q_Simulator:
    def __init__(self, env: Q_Environment, agent: Q_Agent | None):
        self._agent: Q_Agent     = agent
        self._env: Q_Environment = env
        self._gas_coords: set = self._find_gas_coords(env.datapath)

    
    def _find_gas_coords(self, datapaht: Path) -> set:
        """
        Reads every coordinate of the gas plume and returns the plume locations as a set.
        This is a slow operation so the set is written to a pickle file located at ./sim/plume_map/<file_name>_depth_<depth>.pkl
        """
        if datapaht.parent != Path("sim"):
            raise ValueError(r"datapaths has to be on the form ./sim/<SMART_AUVs_*>")
        
        # Check if the plume data exists
        plume_map_path = datapaht.parent / "plume_map" / datapaht.stem
        plume_map_path = Path(f'{plume_map_path}_depth_{self._env.depth}.pkl')
        
        if plume_map_path.exists():
            with open(plume_map_path, "rb") as plume_map:
                return pickle.load(plume_map)
            
        gas_coords = set()        
        # Setting up a progress bar
        coords_list = list(self._env.traverse_environment)
        total_coords = len(coords_list)
        
        # magic formatting, thanks chatGPT
        red_format = '\033[91m{l_bar}{bar}\033[0m [elapsed: {elapsed} remaining: {remaining}]'

        with tqdm(total=total_coords, ncols=100, desc='Processing Coordinates', bar_format=red_format, colour='green') as pbar:
            for idx, coords in enumerate(coords_list):
                gas_val = self._env.get_pH_at_position(coords, 0)
                if self._env.classify_pH(gas_val) != PH_Reading.HIGH:
                    gas_coords.add(coords)
                pbar.update(1)

        with open(plume_map_path, "wb") as plume_map:
            pickle.dump(gas_coords, plume_map)
        return gas_coords
    
    def test_agent(self, max_steps=2000):
        agent.run(max_steps = max_steps)
        print(agent.gas_coords_visited(self._gas_coords))

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, q_table, alpha=0.1, gamma=0.9, epsilon=0.1, temp=1, 
              reward_func=None, policy=None):
        if reward_func is None:
            reward_func = reward_funcs.reward_trace_area
        if policy is None:
            policy = policy_funcs.episilon_greedy

        self._agent = Q_Agent(
            self._env, 
            alpha=alpha, 
            gamma=gamma, 
            epsilon=epsilon, 
            temperature=temp, 
            reward_func=reward_func, 
            policy=policy, 
            start_position=(0, 0, self._env.depth)
        )
        
        # Set the trained Q-table if provided
        if q_table:
            self._agent.q_table = q_table

if __name__ == "__main__":
    env   = Q_Environment(Path(r"./sim/SMART-AUVs_OF-June-1c-0002.nc"), depth=67)
    agent = Q_Agent(env)
    sim = Q_Simulator(env, agent)
    print(sim.test_agent())



