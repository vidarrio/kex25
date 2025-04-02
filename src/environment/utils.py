import random
import numpy as np

# Helper class to implement a simple human planner for a warehouse environment.
# This class computes actions for humans based on their current positions and goals.
# It uses a greedy strategy to minimize the distance to the goal while avoiding obstacles.
# The planner also keeps track of the last position of each human to avoid immediate backtracking.
class SimpleHumanPlanner:
    def __init__(self):
        """
        Initialize the simple human planner
        """
        self.last_positions = {}  # Stores the last position for each human

    def get_actions(self, humans, human_positions, human_goals, grid_size, grid):
        """
        Compute actions for all humans based on their current positions and goals.
        Returns:
            A dictionary mapping each human to an action (0-6).
        """
        human_actions = {}
        for human in humans:
            current = human_positions[human]
            goal = human_goals[human]
            action = self.compute_action(current, goal, human, human_goals=human_goals, grid_size=grid_size, grid=grid)
            human_actions[human] = action
        
            # Update the last position for this human:
            moves = {
                0: (0, -1),    # LEFT: increase row index 
                1: (1, 0),    # UP: increase column index 
                2: (0, 1),   # RIGHT: decrease row index
                3: (-1, 0)    # DOWN: decrease column index
            }
            if action in moves:
                delta = moves[action]
                new_pos = (current[0] + delta[0], current[1] + delta[1])
                self.last_positions[human] = current  # record current position as last
            else:
                self.last_positions[human] = current
        return human_actions

    def compute_action(self, current, goal, human, human_goals, grid_size, grid):
        """
        Compute the next action for a human using a greedy strategy that prioritizes shortest path.
        
        Assumptions:
         - The grid is indexed so that (0,0) is the top-left cell.
         - Row index increases as you move downward.
         - Column index increases as you move to the right.
        
        Strategy:
         1. If human is at goal, assign a new random goal
         2. Try to move directly toward goal (prioritizing biggest distance reduction)
         3. If blocked by shelf, try alternative routes
         4. Only wait if blocked by an agent
        
        Returns:
            An action integer:
              0: Move Left, 1: Move Down, 2: Move Right, 3: Move Up, 6: Wait
              (matches environment's action delta mapping)
        """
        # If at goal, assign a new random goal
        if current == goal:
            assign_new_human_goal(human, human_goals, grid, grid_size)
            goal = human_goals[human]
        
        cur_row, cur_col = current
        goal_row, goal_col = goal
        
        # Calculate distances to goal
        dx = goal_col - cur_col  # horizontal difference
        dy = goal_row - cur_row  # vertical difference

        # Action mapping: direction -> (dr, dc)
        # Matching the environment's actual implementation:
        moves = {
            0: (0, -1),  # LEFT: decrease column
            1: (1, 0),   # DOWN: increase row
            2: (0, 1),   # RIGHT: increase column
            3: (-1, 0)   # UP: decrease row
        }

        # Helper to check if a cell is valid
        def is_valid(pos):
            r, c = pos
            # Out of bounds
            if r < 0 or r >= grid_size[0] or c < 0 or c >= grid_size[1]:
                return 0
            # Cell contains shelf (obstacle)
            if grid[r, c] == 2:
                return 0
            # Cell contains agent (temporary obstacle)
            if grid[r, c] == 1:
                return -1
            # Cell is valid
            return 1

        # Retrieve last position to avoid immediate backtracking
        last_pos = self.last_positions.get(human, None)
        
        # Prioritize moves that reduce distance most
        # First try the direction that reduces the largest distance
        primary_action = None
        secondary_action = None
        
        if abs(dx) > abs(dy):
            # Prioritize horizontal movement
            primary_action = 2 if dx > 0 else 0  # RIGHT(2) if dx positive, LEFT(0) if negative
            secondary_action = 3 if dy < 0 else 1  # UP(3) if dy negative, DOWN(1) if positive
        else:
            # Prioritize vertical movement
            primary_action = 3 if dy < 0 else 1  # UP(3) if dy negative, DOWN(1) if positive
            secondary_action = 2 if dx > 0 else 0  # RIGHT(2) if dx positive, LEFT(0) if negative
        
        # Try primary action first
        if primary_action is not None:
            dr, dc = moves[primary_action]
            new_pos = (cur_row + dr, cur_col + dc)
            validity = is_valid(new_pos)
            
            # If valid and not backtracking
            if validity == 1 and (last_pos is None or new_pos != last_pos):
                return primary_action
            # If blocked by agent, wait
            elif validity == -1:
                return 6
        
        # Try secondary action next
        if secondary_action is not None:
            dr, dc = moves[secondary_action]
            new_pos = (cur_row + dr, cur_col + dc)
            validity = is_valid(new_pos)
            
            # If valid and not backtracking
            if validity == 1 and (last_pos is None or new_pos != last_pos):
                return secondary_action
            # If blocked by agent, wait
            elif validity == -1:
                return 6
        
        # Try all other directions if both primary and secondary failed
        other_actions = [a for a in range(4) if a != primary_action and a != secondary_action]
        for action in other_actions:
            dr, dc = moves[action]
            new_pos = (cur_row + dr, cur_col + dc)
            validity = is_valid(new_pos)
            
            # If valid and not backtracking
            if validity == 1 and (last_pos is None or new_pos != last_pos):
                return action
            # Continue trying other actions even if blocked by agent
        
        print(f"Human {human} is blocked in all directions. Waiting.")
        
        # If no valid move is found, wait
        return 6

def get_random_empty_position(grid, grid_size, avoid_values=[1, 2, 3, 4, 5]):
    """
    Get a random empty position on the grid, avoiding certain cell types
    """
    while True:
        r = random.randint(0, grid_size[0] - 1)
        c = random.randint(0, grid_size[1] - 1)

        if grid[r, c] not in avoid_values:
            return (r, c)
        
def assign_new_human_goal(human, human_goals, grid, grid_size):
    """
    Assign a new random goal to the human.
    The new goal is selected from an empty cell on the grid, avoiding shelves (2)
    and other humans (3). 
    """
    # Allow cells that are not shelves (2)
    human_goals[human] = get_random_empty_position(grid, grid_size, avoid_values=[2])
