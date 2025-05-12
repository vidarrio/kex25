from dataclasses import dataclass
from enum import Enum


@dataclass
class Reward:
    step_penalty: float
    correct_direction_reward: float
    collision_penalty: float
    wrong_pickup_penalty: float
    wrong_dropoff_penalty: float
    oscillation_penalty: float
    task_reward: float


class CellType(Enum):
    EMPTY = 0
    AGENT = 1
    SHELF = 2
    DYNAMIC_OBSTACLE = 3
    PICKUP_POINT = 4
    DROPOFF_POINT = 5

    def __str__(self):
        return self.name.lower()

class ActionType(Enum):
    # Return the action space for a specific agent
    # The action space is discrete with the following actions:
    # 0: Move Left (decrease column)
    # 1: Move Down (increase row)
    # 2: Move Right (increase column)
    # 3: Move Up (decrease row)
    # 4: Pickup Item
    # 5: Drop Item
    # 6: Wait

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    
    PICKUP = 4
    DROP = 5
    WAIT = 6

    def is_movement(self) -> bool:
        if self in [ActionType.LEFT, ActionType.DOWN, ActionType.RIGHT, ActionType.UP]:
            return True
        return False
    
    def get_transpose(self):
        if self == ActionType.LEFT:
            return (0, -1)
        elif self == ActionType.DOWN:
            return (1, 0)
        elif self == ActionType.RIGHT:
            return (0, 1)
        elif self == ActionType.UP:
            return (-1, 0)
        
        raise ValueError(f"Action {self} is not a movement action.")