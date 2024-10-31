import numpy as np

def soft_max(agent, state: tuple[int, int, int]) -> int:
    q_values: np.ndarray = agent.q_table[state]
    max_q_value: np.float64 = np.max(q_values)  
    q_values: np.ndarray = q_values - max_q_value  # Normalize Q-values to prevent overflow

    exponential:np.array = np.exp(q_values / agent._temperature)
    probabilities:np.array = exponential / np.sum(exponential)

    return np.random.choice(3, p=probabilities)

def episilon_greedy(agent, state:tuple[int, int, int]) -> int:
    if np.random.rand() < agent.epsilon:
        # There are 3 possible actions. 
        return np.random.choice(3)
    return np.argmax(agent.q_table[state])