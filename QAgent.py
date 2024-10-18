# %%

import numpy as np
from scipy.spatial import Delaunay

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
        # Assuming env.get_readings returns (right_reading, left_reading, front_reading)
        right_reading, left_reading, front_reading = env.get_readings()
        return (right_reading, left_reading, front_reading)

    def execute_action(self, env, action):
        """Execute the chosen action in the environment and get the feedback."""
        # This will be a placeholder function. Replace it with actual environment interaction.
        next_state, reward = env.perform_action(action)
        return next_state, reward

    def train(self, env, episodes=1000, max_steps_per_episode=100):
        for episode in range(episodes):
            state = self.get_state_from_env(env)
            for step in range(max_steps_per_episode):
                action = self.choose_action(state)
                next_state, reward = self.execute_action(env, action)
                self.update_q_table(state, action, reward, next_state)
                if env.is_done():  # Check if episode is finished
                    break
                state = next_state
            print(f"Episode {episode+1}/{episodes} completed.")

# Example usage
class SimulatedEnvironment:
    def __init__(self):
        # Initialize the environment, if needed
        pass
    
    def get_readings(self):
        # Mock-up function, should return state readings from the environment
        return np.random.choice([0, 1, 2], size=3)

    def perform_action(self, action):
        # Mock-up function, should execute the action and return next state and reward
        next_state = np.random.choice([0, 1, 2], size=3)
        reward = np.random.rand()  # Reward can be customized based on the action and next state
        return next_state, reward
    
    def is_done(self):
        # Mock-up function, should determine if the episode is finished
        return np.random.rand() < 0.1

# Assuming the SimulatedEnvironment interacts internally with the provided functions
env = SimulatedEnvironment()
agent = QAgent()
agent.train(env)


