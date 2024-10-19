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
        # This will be a placeholder function. Replace it with actual environment interaction.
        # For example, interpolate_values might be used to get chemical readings.
        # Assuming env.get_reading_levels returns (right_reading, left_reading, front_reading)
        right_reading, left_reading, front_reading = env.get_reading_levels()
        return (right_reading, left_reading, front_reading)

    def execute_action(self, env, action):
        """Execute the chosen action in the environment and get the feedback."""
        # This will be a placeholder function. Replace it with actual environment interaction.
        next_state, reward = env.perform_action(action)
        return next_state, reward

    def train(self, env, episodes=1000, max_steps_per_episode=100):
        for episode in range(episodes):
            start = perf_counter()
            state = self.get_state_from_env(env)
            min_pH = float('inf')  # Initialize minimum pH tracker
            
            for step in range(max_steps_per_episode):
                # Extract pH readings from the state (assuming the state contains discrete pH states)
                right_pH, left_pH, front_pH = env.get_reading_levels()

                # Update the minimum pH encountered
                min_pH = min(min_pH, right_pH, left_pH, front_pH)

                # Choose and execute the action
                action = self.choose_action(state)
                next_state, reward = self.execute_action(env, action)
                self.update_q_table(state, action, reward, next_state)
                
                # Check if the episode is finished
                if env.is_done():
                    break
                state = next_state
            
            # At the end of the episode, print the minimum pH found
            print(f"Episode {episode+1}/{episodes} completed. Minimum pH found: {min_pH:.2f}")
            print(f"Time to complete episode: {perf_counter() - start:.2f} seconds.")

# Example usage
class SimulatedEnvironment:
    def __init__(self, chem_data_path, x_start, y_start, z_start=0):
        self.chemical_dataset = chem_utils.load_chemical_dataset(chem_data_path)
        
        # Initialize the agent's position and heading
        self.x = x_start
        self.y = y_start
        self.z = z_start  # Default depth or height, if applicable
        self.heading = 'north'  # Initial heading of the agent (can be north, south, east, west)

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
        front_pH = self._get_pH_at_position(front_pos, time_target)
        left_pH = self._get_pH_at_position(left_pos, time_target)
        right_pH = self._get_pH_at_position(right_pos, time_target)

        return front_pH, left_pH, right_pH

    def get_reading_levels(self):
        """
        Return pH readings in the surroundings based on the agent's current heading.
        - 0 if pH > 7.77 (basic)
        - 2 if pH < 7.5 (acidic)
        - 1 if 7.5 <= pH <= 7.77 (neutral)
        The readings are taken for the front, left, and right directions, based on the agent's heading.
        """
        front_pH, left_pH, right_pH = self.get_current_pH_values()

        # Classify the pH values into discrete states
        front_state = self._classify_pH(front_pH)
        left_state = self._classify_pH(left_pH)
        right_state = self._classify_pH(right_pH)

        return front_state, left_state, right_state

    def _get_pH_at_position(self, position, time_target):
        """
        Extract the pH value from the dataset at a specific position (x, y, z) and time.
        """
        x_target, y_target, z_target = position

        # Metadata includes the target coordinates and the search steps
        metadata = (x_target, y_target, z_target, time_target, 1)
        
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
        """
        if pH_value > 7.77:
            return 0
        elif pH_value < 7.5:
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
        reward = np.random.rand()  # Define own reward function later
        return next_state, reward

    def _move_forward(self):
        """
        Move forward one step in the direction the agent is currently facing.
        """
        # Move based on the current heading
        if self.heading == 'north':
            if self.y + 1 < 250:
                self.y += 1  # Move up in y-axis
        elif self.heading == 'east':
            if self.x + 1 < 250:
                self.x += 1  # Move right in x-axis
        elif self.heading == 'south':
            if self.x > 0:
                self.y -= 1  # Move down in y-axis
        elif self.heading == 'west':
            if self.y > 0:
                self.x -= 1  # Move left in x-axis

    def is_done(self):
        # To be determined
        return False

# Assuming the SimulatedEnvironment interacts internally with the provided functions
env = SimulatedEnvironment(f"../1c_co2_medium/SMART-AUVs_OF-June-1c-0001.nc", np.random.randint(0, 250), np.random.randint(0, 250), 69)
agent = QAgent()
agent.train(env, episodes=100)


