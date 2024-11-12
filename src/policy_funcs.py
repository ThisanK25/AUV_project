import numpy as np

def soft_max(agent, state: tuple[int, int, int]) -> int:
    q_values: np.ndarray = agent.q_table[state]
    # Softmax is not numerically stable, so this will normalize the exponent to prevent overflow
    max_q_value: np.float64 = np.max(q_values)  
    q_values: np.ndarray = q_values - max_q_value

    exponential:np.array = np.exp(q_values / agent._temperature)
    probabilities:np.array = exponential / np.sum(exponential)
    # p is the probability distribution of the possible choises
    return np.random.choice(3, p=probabilities)

def episilon_greedy(agent, state:tuple[int, int, int]) -> int:
    if np.random.rand() < agent.epsilon:
        # There are 3 possible actions. 
        return np.random.choice(3)
    return np.argmax(agent.q_table[state])