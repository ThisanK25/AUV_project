from enum import Enum
class Direction(Enum):
    North = 0
    East  = 1
    South = 2
    West  = 3

class PH_Reading(Enum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2

class AUV_ACTIONS(Enum):
    RIGHT = 0
    LEFT = 1
    FORWARD = 2


if __name__ == "__main__":
    print(list(Direction))
