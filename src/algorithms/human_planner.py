class SimpleHumanPlanner:
    def __init__(self, env):
        """
        Initialize the simple human planner with the warehouse environment.
        """
        self.env = env
        self.last_positions = {}  # Stores the last position for each human


    def get_actions(self):
        """
        Compute actions for all humans based on their current positions and goals.
        Returns:
            A dictionary mapping each human to an action (0-6).
        """
        human_actions = {}
        for human in self.env.humans:
            current = self.env.human_positions[human]
            goal = self.env.human_goals[human]
            action = self.compute_action(current, goal, human)
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

    def compute_action(self, current, goal, human):
        """
        Compute the next action for a human using a greedy strategy that first minimizes the horizontal distance.
        
        Assumptions:
         - The grid is indexed so that (0,0) is the bottom-left cell.
         - Row index increases as you move upward.
         - Column index increases as you move to the right.
        
        Strategy:
         1. If the human is at its goal, immediately assign a new random goal.
         2. If there is a horizontal difference (dx != 0), attempt to move right (if dx > 0) or left (if dx < 0).
         3. If that move is blocked, try a vertical move (if there is a vertical difference).
         4. If still blocked, try the opposite horizontal direction.
         5. If none of these moves are valid, try any vertical move.
         6. If no valid move is available, wait (action 6).
        
        Returns:
            An action integer:
              0: Up, 1: Right, 2: Down, 3: Left, 6: Wait.
        """
        # If at goal, assign a new random goal.
        if current == goal:
            self.env._assign_new_human_goal(human)
            goal = self.env.human_goals[human]
        
        cur_row, cur_col = current
        goal_row, goal_col = goal
        dx = goal_col - cur_col  # horizontal difference
        dy = goal_row - cur_row  # vertical difference

        
        moves = {
            0: (0, -1),    # LEFT: increase row index 
            1: (1, 0),    # UP: increase column index 
            2: (0, 1),   # RIGHT: decrease row index
            3: (-1, 0)    # DOWN: decrease column index
        }

        # Helper to check if a cell is valid (inside grid and not blocked by a shelf or dynamic obstacle).
        def valid(pos):
            r, c = pos
            if r < 0 or r >= self.env.grid_size[0] or c < 0 or c >= self.env.grid_size[1]:
                return False
            if self.env.grid[pos] in [2, 3]:
                return False
            return True

        
        # Retrieve last position if it exists.
        last_pos = self.last_positions.get(human, None)

        candidate = None
        
        print(cur_row, cur_col)
        # Primary strategy: minimize horizontal difference first.
        if dx != 0:
            desired_action = 2 if dx > 0 else 0  # right if dx positive, left if dx negative
            delta = moves[desired_action]
            new_pos = (cur_row + delta[0], cur_col + delta[1])
            if valid(new_pos) and (last_pos is None or new_pos != last_pos):
                candidate = desired_action
            else:
                # Fallback: try vertical move if there is a vertical difference.
                if dy != 0:
                    horizontal_action = 1 if dy > 0 else 3  # up if dy positive, down if dy negative
                    delta = moves[horizontal_action]
                    new_pos = (cur_row + delta[0], cur_col + delta[1])
                    if valid(new_pos) and (last_pos is None or new_pos != last_pos):
                        candidate = horizontal_action
                        
                if candidate is None:
                    # Fallback: try the opposite horizontal direction.
                    opposite_action = 0 if desired_action == 2 else 2
                    delta = moves[opposite_action]
                    new_pos = (cur_row + delta[0], cur_col + delta[1])
                    if valid(new_pos) and (last_pos is None or new_pos != last_pos):
                        candidate = opposite_action
                if candidate is None:
                    # Fallback: try vertical moves in both directions.
                    for vertical_action in [1, 3]:
                        delta = moves[vertical_action]
                        new_pos = (cur_row + delta[0], cur_col + delta[1])
                        if valid(new_pos) and (last_pos is None or new_pos != last_pos):
                            candidate = vertical_action
                            break
            if candidate is not None:
                return candidate
            else:
                return 6
        else:
            # Horizontal difference is 0; minimize vertical difference.
            if dy != 0:
                desired_action = 1 if dy > 0 else 3  # up if dy positive, down if dy negative
                delta = moves[desired_action]
                new_pos = (cur_row + delta[0], cur_col + delta[1])
                if valid(new_pos) and (last_pos is None or new_pos != last_pos):
                    candidate = desired_action
                else:
                    # Try horizontal moves to bypass the obstacle.
                    for horizontal_action in [0, 2]:
                        delta = moves[horizontal_action]
                        new_pos = (cur_row + delta[0], cur_col + delta[1])
                        if valid(new_pos) and (last_pos is None or new_pos != last_pos):
                            candidate = horizontal_action
                            break
                if candidate is not None:
                    return candidate
                else:
                    return 6
            else:
                return 6
