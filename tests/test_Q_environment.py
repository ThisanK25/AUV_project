
import test_header
# Importing modules from src
from QAgent_new import Q_Agent
from Q_environment import Q_Environment
from QAgent_Enums import Direction
from pathlib import Path

def test_classifier(env: Q_Environment, pos: tuple[int, int, int]):
    vals = env.get_current_pH_values((94, 71, 68), Direction.North)
    classified = env._classify_pH(vals[0])
    print(classified)

if __name__ == "__main__":
    env = Q_Environment(Path(r"C:\Users\yllip\Documents\in5490\AUV_project\sim\SMART-AUVs_OF-June-1c-0002.nc"), depth=68)
    test_classifier(env, (0, 0, 0))
