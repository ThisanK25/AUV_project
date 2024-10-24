# %%
import numpy as np
from scipy.spatial import Delaunay
import path
import chem_utils
from time import perf_counter

class QAgent:
    def __init__(self, q_table_shape=(3, 3, 3, 3), alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros(q_table_shape)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.possible_states = np.array([0, 1, 2])  # 0: High, 1: Medium, 2: Low
        self.possible_actions = np.array([0, 1, 2])  # 0: Right, 1: Left, 2: Forward
        
        # ---- Variables for reward -----
        self.found_gas = False
        self.time_steps_in_high = 0
        self.time_steps_in_medium = 0
        self.reward_function = self.reward_gas_level  # Default reward function
        # -------------------------------

    def set_reward_function(self, reward_function):
        self.reward_function = reward_function

    def initialize_state(self):
        """Initialize the agent to a random state."""
        init_state = tuple(np.random.choice(self.possible_states, size=3))
        return init_state

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.possible_actions)  # Exploration
        else:
            state_action_values = self.q_table[state]
            max_value = np.max(state_action_values)
            actions_with_max_value = np.where(state_action_values == max_value)[0]
            return np.random.choice(actions_with_max_value)  # Exploitation

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[state][action] = new_q

    def get_state_from_env(self, env):
        """Extract the current state from the environment."""
        right_reading, left_reading, front_reading = env.get_reading_levels()
        return (right_reading, left_reading, front_reading)

    def execute_action(self, env, action):
        """Execute the chosen action in the environment and get the feedback."""
        next_state = env.perform_action(action)
        return next_state
    
    def train(self, env, episodes=1000, max_steps_per_episode=100):
        for episode in range(episodes):
            start = perf_counter()
            state = self.get_state_from_env(env)
            min_pH = float('inf')  # Initialize minimum pH tracker
            
            for step in range(max_steps_per_episode):
                # Extract pH readings from the state (assuming the state contains discrete pH states)
                right_pH, left_pH, front_pH = env.get_current_pH_values()

                # Update the minimum pH encountered
                min_pH = min(min_pH, right_pH, left_pH, front_pH)

                # Choose and execute the action
                action = self.choose_action(state)
                next_state = self.execute_action(env, action)
                
                # Evaluate reward using the selected reward function
                reward = self.reward_function(next_state)
                
                self.update_q_table(state, action, reward, next_state)
                
                # Check if the episode is finished
                if env.is_done():
                    break
                state = next_state
            
            # At the end of the episode, print the minimum pH found
            print(f"---------------Episode {episode+1}/{episodes} completed.--------------")
            print(f"Minimum pH found: {min_pH:.2f}\n")
            print(" State:         H  M  L                        H  M  L")
            print(f"Current state: {state}          Next state: {next_state}\n")
            print(f"Position: x:{env.x}, y: {env.y}, z: {env.z}")
            print(f"Time to complete episode: {perf_counter() - start:.2f} seconds.")
            print(f"Reward: {reward}\n")
            print("-------------------------------------------------------------")

    # Reward functions encapsulated in the QAgent class
    def reward_gas_level(self, next_state):
        """
        High reward for high gas reading, medium reward for medium reading, negative for low reading.
        """
        min_pH = min(next_state)

        
        if min_pH == 2:  # Very low pH - high gas consentration
            self.found_gas = True
            return 10
        elif min_pH == 0:  # High pH - low gas consentration
            if not self.found_gas:
                return 0
            else:
                return -10
        else:
            self.found_gas = True
            return 5  # Medium gas reading

    def reward_exposure_time(self, next_state):
        """
        Longer exposure to high or medium gas readings increases the reward.
        """
        right, left, front = next_state
        if right == 2 or left == 2 or front == 2:
            self.time_steps_in_high += 1
            return self.time_steps_in_high
        elif right == 1 or left == 1 or front == 1:
            self.time_steps_in_medium += 1
            return self.time_steps_in_medium
        return -1  # Penalty for low exposure

    def reward_plume_field(self, next_state):
        """
        Negative reward if agent exits plume, positive if in plume, and higher if gas reading gets higher.
        """
        right, left, front = next_state
        min_pH = min(next_state)
        if right == 0 and left == 0 and front == 0:
            return -10  # Outside plume
        reward = 0
        # Adjust reward based on gas readings
        if min_pH == 2:
            reward += 10  # Strongest gas reading
        elif min_pH == 1:
            reward += 5   # Medium gas reading
        return reward
    
# Example usage
class Environment_interaction:
    def __init__(self, chem_data_path, x_start, y_start, z_start=0, confined=False, x_bounds=(0, 250), y_bounds=(0, 250)):
        self.chemical_dataset = chem_utils.load_chemical_dataset(chem_data_path)
        
        # Initialize the agent's position and heading
        self.x = x_start
        self.y = y_start
        self.z = z_start                    # Default depth or height, if applicable
        self.radius_of_gas_reading = 5      # The radius of the area that the agent collects readings from
        self.heading = 'north'              # Initial heading of the agent (can be north, south, east, west)
        self.confined = confined            # Whether the agent is confined to a specific area
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds

    def get_current_pH_values(self):
        """
        Return the actual pH values for the agent's front, left, and right positions (not the classified states).
        """
        time_target = 0  # Example: Use the first time step
        steps = 1  # Coordinate steps

        # Define positions relative to the agent's current heading
        if self.heading == 'north':
            front_pos = (self.x, self.y + steps, self.z)
            left_pos = (self.x - steps, self.y, self.z)
            right_pos = (self.x + steps, self.y, self.z)
        elif self.heading == 'east':
            front_pos = (self.x + steps, self.y, self.z)
            left_pos = (self.x, self.y + steps, self.z)
            right_pos = (self.x, self.y - steps, self.z)
        elif self.heading == 'south':
            front_pos = (self.x, self.y - steps, self.z)
            left_pos = (self.x + steps, self.y, self.z)
            right_pos = (self.x - steps, self.y, self.z)
        elif self.heading == 'west':
            front_pos = (self.x - steps, self.y, self.z)
            left_pos = (self.x, self.y - steps, self.z)
            right_pos = (self.x, self.y + steps, self.z)

        # Extract actual pH values (not classified)
        right_pH = self._get_pH_at_position(right_pos, time_target)
        left_pH = self._get_pH_at_position(left_pos, time_target)
        front_pH = self._get_pH_at_position(front_pos, time_target)

        return right_pH, left_pH, front_pH

    def get_reading_levels(self):
        """
        Return pH readings in the surroundings based on the agent's current heading.
        - 0 if pH > 7.77 (basic)
        - 2 if pH < 7.5 (acidic)
        - 1 if 7.5 <= pH <= 7.77 (neutral)
        The readings are taken for the front, left, and right directions, based on the agent's heading.
        """
        right_pH, left_pH, front_pH = self.get_current_pH_values()

        # Classify the pH values into discrete states
        right_state = self._classify_pH(right_pH)
        left_state = self._classify_pH(left_pH)
        front_state = self._classify_pH(front_pH)

        return right_state, left_state, front_state

    def _get_pH_at_position(self, position, time_target):
        """
        Extract the pH value from the dataset at a specific position (x, y, z) and time.
        """
        x_target, y_target, z_target = position

        # Metadata includes the target coordinates and the search steps
        metadata = (x_target, y_target, z_target, time_target, self.radius_of_gas_reading)
        
        # Extract the pH value from the dataset using the provided function
        average_pH, _ = chem_utils.extract_chemical_data_for_volume(
            self.chemical_dataset, metadata, data_variable='pH')

        return average_pH

    def _classify_pH(self, pH_value):
        """
        Classify pH value into discrete states:
        - Return 0 if pH > 7.77 (basic)
        - Return 2 if pH < 7.5 (acidic)
        - Return 1 if 7.5 <= pH <= 7.77 (neutral)
        - This is basic settings for depth: 69 meter
        """
        ph_69 = [7.7, 7.5] 
        ph_68 = [7.74, 7.62] 
        ph_67 = [7.76, 7.69] 
        ph_66 = [7.77, 7.745] 
        ph_65 = [7.772, 7.755] 
        ph_64 = [7.773, 7.766]
        
        current_pH_classification_limits = []
        
        match self.z:
            case 69:
                current_pH_classification_limits = ph_69
            case 68:
                current_pH_classification_limits = ph_68
            case 67:
                current_pH_classification_limits = ph_67
            case 66:
                current_pH_classification_limits = ph_66
            case 65:
                current_pH_classification_limits = ph_65
            case 64:
                current_pH_classification_limits = ph_64

        high = current_pH_classification_limits[0]
        low = current_pH_classification_limits[1]
        
        if pH_value > high:
            return 0
        elif pH_value < low:
            return 2
        else:
            return 1

    def perform_action(self, action):
        """
        Perform the given action (move forward, left, or right) relative to the agent's current heading.
        """
        directions = ['north', 'east', 'south', 'west']

        if action == 2:  # Move forward
            self._move_forward()

        elif action == 0:  # Move right
            # Turn right relative to the current heading
            current_index = directions.index(self.heading)
            self.heading = directions[(current_index + 1) % 4]  # Update heading
            self._move_forward()

        elif action == 1:  # Move left
            # Turn left relative to the current heading
            current_index = directions.index(self.heading)
            self.heading = directions[(current_index - 1) % 4]  # Update heading
            self._move_forward()

        # Get next state readings and assign a reward
        next_state = self.get_reading_levels()
        return next_state

    # def _move_forward(self):
    #     """
    #     Move forward one step in the direction the agent is currently facing.
    #     """
    #     # Move based on the current heading
    #     if self.heading == 'north':
    #         if self.y + 1 < 250:
    #             self.y += 1  # Move up in y-axis
    #     elif self.heading == 'east':
    #         if self.x + 1 < 250:
    #             self.x += 1  # Move right in x-axis
    #     elif self.heading == 'south':
    #         if self.y > 0:
    #             self.y -= 1  # Move down in y-axis
    #     elif self.heading == 'west':
    #         if self.x > 0:
    #             self.x -= 1  # Move left in x-axis

    def _move_forward(self):
        if self.heading == 'north' and (not self.confined or self.y + 1 <= self.y_max):
            self.y += 1  # Move up in y-axis
        elif self.heading == 'east' and (not self.confined or self.x + 1 <= self.x_max):
            self.x += 1  # Move right in x-axis
        elif self.heading == 'south' and (not self.confined or self.y - 1 >= self.y_min):
            self.y -= 1  # Move down in y-axis
        elif self.heading == 'west' and (not self.confined or self.x - 1 >= self.x_min):
            self.x -= 1  # Move left in x-axis

    def is_done(self):
        # To be determined
        return False
    

# Simulation setup

x_start = np.random.randint(0, 250)
y_start = np.random.randint(0, 250)
z_start = 68

# Set boundaries for the confined environment
x_bounds = (100, 250)
y_bounds = (100, 200)

env = Environment_interaction(f"../SMART-AUVs_OF-June-1c-0002.nc", x_start, y_start, z_start)
conf_env = Environment_interaction("../SMART-AUVs_OF-June-1c-0002.nc", x_start, y_start, z_start, confined=True, x_bounds=x_bounds, y_bounds=y_bounds)


# %%
print("Training with confined environment")
agent = QAgent()
agent.set_reward_function(agent.reward_gas_level)
print("Training with reward function 1: High gas reading reward")
agent.train(conf_env, episodes=100)


# %%
# High reward for high gas reading, medium reward for medium reading, negative for low reading
agent = QAgent()
agent.set_reward_function(agent.reward_gas_level)
print("Training with reward function 1: High gas reading reward")
agent.train(env, episodes=100)

# %%
# The longer the agent is exposed to high or medium gas reading, the higher the reward
agent = QAgent()
agent.set_reward_function(agent.reward_exposure_time)
print("Training with reward function 2: Exposure time reward")
agent.train(env, episodes=100)

# %%
# If the agent exits the field of plume the reward is negative, if it is in the plume it is positive, and if the gas reading gets higher (or lower pH-value) the reward gets higher
agent = QAgent()
agent.set_reward_function(agent.reward_plume_field)
print("Training with reward function 3: Plume field and gas reading reward")
agent.train(env, episodes=100)
