from QAgent_Enums import PH_Reading
import numpy as np

def reward_gas_level(_, next_state) -> int:
    """
    High reward for high gas reading, medium reward for medium reading, negative for low reading.
    """
    min_pH = PH_Reading(min(next_state))
    if min_pH == PH_Reading.LOW:  # Very low pH - high gas concentration
        return 10
    elif min_pH == PH_Reading.HIGH:  # High pH - low gas concentration
        return -1
    else:
        return 5  # Medium gas reading

def reward_exposure_time(agent, next_state) -> int:
    """
    Longer exposure to high or medium gas readings increases the reward.
    """
    max_val = np.max(next_state)
    if max_val == PH_Reading.LOW:
        agent.time_steps_in_high += 1
        return agent.time_steps_in_high
    elif max_val == PH_Reading.MEDIUM:
        agent.time_steps_in_medium += 1
        return agent.time_steps_in_medium
    return -1  # Penalty for low exposure


def reward_trace_area(agent, next_state:tuple[PH_Reading, PH_Reading, PH_Reading]) -> int:
    """
    Positive reward if the agent is on a state boundery
    """
    left, right, forward = next_state
    if left == forward == right:
        if left == PH_Reading.HIGH:
            return -3
        return -1
    if left != PH_Reading.HIGH and right == PH_Reading.HIGH:
        return 5
    return 1

def reward_plume_field(agent, next_state)->int:
    """
    Negative reward if agent exits plume, positive if in plume, and higher if gas reading gets higher.
    """
    if agent.visited():
        return -5
    min_val = np.min(next_state)
    min_pH = min(next_state)
    if min_val == PH_Reading.HIGH:
        return -1  # Outside plume
    reward = 0
    # Adjust reward based on gas readings
    if min_pH == PH_Reading.LOW:
        reward += 2  # Strongest gas reading
    elif min_pH == PH_Reading.MEDIUM:
        reward += 1   # Medium gas reading
    return reward