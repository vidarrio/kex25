import random
import numpy as np

# Helper class to implement a simple human planner for a warehouse environment.
# This class computes actions for humans based on their current positions and goals.
# It uses a greedy strategy to minimize the distance to the goal while avoiding obstacles.
# The planner also keeps track of the last position of each human to avoid immediate backtracking.
class SimpleHumanPlanner:
    def __init__(self):
        """
        Initialize the simple human planner with position history
        """
        self.last_positions = {}  # Stores the last position for each human
        self.position_history = {}  # Stores recent positions to detect cycles
        self.history_length = 8  # How many recent positions to track for cycle detection

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
            
            # Initialize position history for this human if needed
            if human not in self.position_history:
                self.position_history[human] = []
            
            # Check if we're in a cycle (same position visited multiple times)
            if self._is_in_cycle(human, current):
                # Break the cycle by assigning a new random goal
                # print(f"Human {human} detected in a cycle. Assigning new goal.")
                assign_new_human_goal(human, human_goals, grid, grid_size)
                goal = human_goals[human]
                # Clear the position history to start fresh
                self.position_history[human] = []
            
            action = self.compute_action(current, goal, human, human_goals, grid_size, grid)
            human_actions[human] = action
            
            # Update the position history
            self._update_position_history(human, current)
            
            # Update the last position for this human:
            moves = {
                0: (0, -1),  # LEFT
                1: (1, 0),   # DOWN
                2: (0, 1),   # RIGHT 
                3: (-1, 0)   # UP
            }
            if action in moves:
                delta = moves[action]
                new_pos = (current[0] + delta[0], current[1] + delta[1])
                self.last_positions[human] = current  # record current position as last
            else:
                self.last_positions[human] = current
                
        return human_actions
    
    def _update_position_history(self, human, position):
        """
        Update the position history for a human
        """
        self.position_history[human].append(position)
        # Keep only the most recent positions
        if len(self.position_history[human]) > self.history_length:
            self.position_history[human].pop(0)
    
    def _is_in_cycle(self, human, position):
        """
        Check if a human is stuck in a movement cycle
        Only detect true movement cycles, not waiting patterns
        """
        history = self.position_history.get(human, [])
        
        # Need at least 6 positions to detect a real cycle
        if len(history) < 6:
            return False
        
        # Check if we've been at the same position repeatedly with movement in between
        # This indicates a cycle rather than waiting
        if position in history[:-1]:  # Check if current position was seen before
            # Find indices where this position appears
            indices = [i for i, pos in enumerate(history) if pos == position]
            
            if len(indices) >= 2:
                # Check if there was movement between the occurrences
                # Extract the positions between previous occurrence and now
                last_idx = indices[-2]  # Second-to-last occurrence
                between_positions = history[last_idx+1:-1]
                
                # If there were at least 2 different positions between occurrences,
                # this is a true cycle, not just waiting in place
                unique_between = set(between_positions)
                if len(unique_between) >= 2 and position in unique_between:
                    return True
        
        # Detect repeating patterns (rectangular movement)
        if len(history) >= 8:
            # Check for a 4-step movement cycle that repeats
            # But only if the pattern contains different positions (true movement)
            pattern = history[-4:]
            if pattern == history[-8:-4] and len(set(pattern)) > 1:
                return True
        
        return False

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
