from collections import deque
import heapq
import numpy as np
import time

DEBUG_NONE = 0
DEBUG_CRITICAL = 1
DEBUG_INFO = 2
DEBUG_VERBOSE = 3
DEBUG_ALL = 4
DEBUG_SPECIFIC = 5

class AStarAgent:
    """
    A* agent implementation for warehouse environment.
    Uses centralized planning and decentralized execution.
    """

    def __init__(self, env, debug_level=1, global_deadlock_threshold=5, local_deadlock_threshold=3):
        """Initialize A* agent with the warehouse environment."""
        self.env = env

        # Store pre-computed paths for each agent
        self.paths = {}
        # Track current step in the path for each agent
        self.path_indices = {}
        # Track whether agents need path re-planning
        self.need_replanning = {}
        # Deadlock detection
        self.consecutive_waits = {agent: 0 for agent in self.env.agents}
        self.global_deadlock_threshold = global_deadlock_threshold  # Max global waits before triggering deadlock (any position)
        self.local_deadlock_threshold = local_deadlock_threshold  # Max local waits before triggering replanning (specific position)
        # Position history for oscillation detection
        self.position_history = {agent: [] for agent in self.env.agents}
        self.oscillation_detection_threshold = 6  # Need at least 6 positions to detect oscillation of length 2

        # Debug level
        self.debug_level = debug_level

        # Action mapping for movement
        self.action_to_delta = {
            0: (0, -1),  # LEFT: decrease column
            1: (1, 0),   # DOWN: increase row
            2: (0, 1),   # RIGHT: increase column
            3: (-1, 0),  # UP: decrease row
            4: None,     # Pickup
            5: None,     # Dropoff
            6: None,     # Wait
        }

        # Delta to action mapping 
        self.delta_to_action = {
            (0, -1): 0,  # LEFT
            (1, 0): 1,   # DOWN
            (0, 1): 2,   # RIGHT
            (-1, 0): 3,  # UP
        }

    def debug(self, level, message):
        """Print debug message if current debug level is high enough."""
        if self.debug_level >= level or level == DEBUG_SPECIFIC:
            print(message)

    def plan_all_paths(self):
        """Plan paths for all agents with collision avoidance."""
        # Reset paths and indices
        self.paths = {}
        self.path_indices = {agent: 0 for agent in self.env.agents}
        self.need_replanning = {agent: False for agent in self.env.agents}

        # Get global state for planning
        global_state, additional_info = self.env.get_global_state()

        # Extract obstacle map from the global state (channel 1 is shelves)
        # First reshape global_state back to its multi-channel grid form
        channels = 8  # Based on your current implementation
        height, width = self.env.grid_size
        reshaped_state = global_state.reshape(channels, height, width)
        
        # Extract obstacle map (shelves)
        obstacle_map = reshaped_state[1].copy()  # Channel 1 is shelves

        # Debug print obstacle map shape
        self.debug(DEBUG_VERBOSE, f"Obstacle map shape: {obstacle_map.shape}, Grid size: {self.env.grid_size}")
        self.debug(DEBUG_VERBOSE, f"Transposed shape would be: {obstacle_map.T.shape}")

        # Dictionary to track space-time reservations for collision avoidance
        reservations = {}

        # Debug
        self.debug(DEBUG_INFO, "\n===== Planning Paths =====")

        # Get planning priority order
        priority_order = self.get_priority_order()
        
        # Plan paths for each agent in priority order
        for agent in priority_order:
            # Get agent's current position, goal, and carrying status
            start = self.env.agent_positions[agent]
            goal = self.env.agent_goals[agent]
            carrying = self.env.agent_carrying[agent]

            self.debug(DEBUG_INFO, f"{agent}: Start={start}, Goal={goal}, Carrying={carrying}")

            # Plan path avoiding obstacles and other agents
            path = self._compute_path(start, goal, obstacle_map, reservations, agent)

            # Debug
            if path:
                self.debug(DEBUG_INFO, f"{agent}: Found path of length {len(path)}")
                self.debug(DEBUG_VERBOSE, f"Path: {path[:5]}... (showing first 5 steps)")
            else:
                self.debug(DEBUG_INFO, f"{agent}: No path found!")

            # Plan to pickup / dropoff
            if path:  # If a path was found
                # Check if the last action in the path is already a pickup/dropoff action
                if len(path) > 0 and path[-1][2] in [4, 5]:
                    # Path already ends with a pickup/dropoff action, no need to add it
                    pass
                else:
                    # Add pickup/dropoff action if needed
                    if not carrying and goal in self.env.pickup_points:
                        path.append((goal[0], goal[1], 4))  # Append pickup action
                        self.debug(DEBUG_VERBOSE, f"{agent}: Added pickup action at goal")
                    elif carrying and goal in self.env.dropoff_points:
                        path.append((goal[0], goal[1], 5))  # Append dropoff action
                        self.debug(DEBUG_VERBOSE, f"{agent}: Added dropoff action at goal")

                # Store the path
                self.paths[agent] = path

                # Reserve space-time points for this agent's path
                time_step = 0
                for pos_r, pos_c, action in path:
                    # Reserve space-time points up to a maximum reservation horizon
                    max_reservation_horizon = 15
                    if time_step < max_reservation_horizon:
                        reservations[(pos_r, pos_c, time_step)] = agent
                        
                        # For the last position, reserve it for all future time steps
                        # to prevent other agents from moving there
                        if time_step == len(path) - 1:
                            for future_time in range(time_step + 1, max_reservation_horizon):
                                reservations[(pos_r, pos_c, future_time)] = agent
                    time_step += 1
            else:
                # If no path found, agent will wait
                self.paths[agent] = [(start[0], start[1], 6)]  # Wait action
                self.debug(DEBUG_INFO, f"{agent}: No path found, waiting")

        return self.paths

    def get_actions(self):
        """Get actions for all agents based on their planned paths."""
        actions = {}
        
        # First, check if any agent has waited too long (deadlock)
        for agent in self.env.agents:
            if agent in self.paths and self.path_indices[agent] < len(self.paths[agent]):
                _, _, action = self.paths[agent][self.path_indices[agent]]

                # Track consecutive waits for deadlock detection
                if action == 6:  # Wait action
                    self.consecutive_waits[agent] += 1
                    if self.consecutive_waits[agent] >= self.global_deadlock_threshold:
                        self.debug(DEBUG_CRITICAL, f"Deadlock detected for agent {agent}, has waited for {self.consecutive_waits[agent]} steps!")
                        self.need_replanning[agent] = True
                        self.consecutive_waits[agent] = 0
                else:
                    # Reset wait count if action is not wait
                    self.consecutive_waits[agent] = 0

        # Get actions for each agent from their path
        for agent in self.env.agents:
            if agent in self.paths and self.path_indices[agent] < len(self.paths[agent]):
                # Get the next action from the path
                _, _, action = self.paths[agent][self.path_indices[agent]]
                actions[agent] = action
                self.path_indices[agent] += 1

                # Check if the agent has reached the end of its path
                if self.path_indices[agent] >= len(self.paths[agent]):
                    # If the agent has reached its goal, mark it as needing replanning
                    self.need_replanning[agent] = True
            else:
                # Default to wait if no path or path is exhausted
                actions[agent] = 6  # Wait
                self.need_replanning[agent] = True

        # Replan if necessary
        replanning_needed = any(self.need_replanning.values())
        if replanning_needed:
            self.plan_all_paths()
            
        # Return the actions
        return actions
    
    def get_priority_order(self):
        """
        Get the priority order of agents based on their current state.
        
        Returns:
            List of agents in priority order
        """
        priority_order = []

        # Carrying agents first
        for agent in self.env.agents:
            if self.env.agent_carrying[agent]:
                priority_order.append(agent)

        # Agents not at their goals
        for agent in self.env.agents:
            if agent not in priority_order and self.env.agent_positions[agent] != self.env.agent_goals[agent]:
                priority_order.append(agent)

        # Lastly, agents at their goals
        for agent in self.env.agents:
            if agent not in priority_order:
                priority_order.append(agent)

        return priority_order

    def _compute_path(self, start, goal, obstacle_map, reservations, agent):
        """
        Compute a path from start to goal using A* algorithm.
        
        Args:
            start: (row, col) start position
            goal: (row, col) goal position
            obstacle_map: Binary map where 1 indicates an obstacle
            reservations: Dictionary of space-time reservations {(row, col, t): agent_id}
            agent: Current agent ID

        Returns:
            List of (row, col, action) tuples representing the path
        """
        # Check if agent is already at the goal
        if start == goal:
            # When already at goal, create a path with the appropriate action
            carrying = self.env.agent_carrying[agent]
            if not carrying and goal in self.env.pickup_points:
                return [(goal[0], goal[1], 4)]  # Return path with pickup action
            elif carrying and goal in self.env.dropoff_points:
                return [(goal[0], goal[1], 5)]  # Return path with dropoff action
            else:
                return [(goal[0], goal[1], 6)]  # Just wait if no action needed
            
        # Priority queue for A* search [(f_score, counter, (row, col, t))]
        counter = 0
        open_set = []

        # Start position with time=0
        start_node = (start[0], start[1], 0)  # (row, col, time)

        # Track g_score (cost from start to node)
        g_score = {start_node: 0}

        # Track f_score (estimated cost from start to goal)
        f_score = {start_node: self._heuristic(start, goal)}

        # Add start to open set
        heapq.heappush(open_set, (f_score[start_node], counter, start_node))
        counter += 1

        # Track where each node came from
        came_from = {}

        # Maximum time steps to search (prevent infinite loops)
        max_time_steps = 100
        
        # For detecting impossible paths
        explored_nodes = 0
        max_explored = 10000  # Limit exploration to prevent hanging

        # Debug
        self.debug(DEBUG_INFO, f"  Starting A* search from {start} to {goal}")

        while open_set and explored_nodes < max_explored:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            current_r, current_c, current_t = current

            explored_nodes += 1
            
            # Periodically print progress
            if explored_nodes % 1000 == 0:
                self.debug(DEBUG_VERBOSE, f"  Explored {explored_nodes} nodes, current: {(current_r, current_c)}")

            # Stop if we've exceeded max time steps
            if current_t > max_time_steps:
                self.debug(DEBUG_INFO, f"  Path search exceeded max time steps for {agent}")
                break

            # Check if we've reached the goal
            if (current_r, current_c) == goal:
                path = self._reconstruct_path(came_from, current)
                self.debug(DEBUG_INFO, f"  Found path of length {len(path)} after exploring {explored_nodes} nodes")
                return path

            # Explore neighbors (4 connected grid)
            for action in range(4):  # Only movement actions (0-3)
                delta = self.action_to_delta[action]
                if delta is None:
                    continue
                    
                dr, dc = delta
                next_r, next_c = current_r + dr, current_c + dc
                next_t = current_t + 1
                next_node = (next_r, next_c, next_t)

                # Check if position is valid
                if (
                    next_r < 0 or next_r >= self.env.grid_size[0] or
                    next_c < 0 or next_c >= self.env.grid_size[1]
                ):
                    continue

                # Check if the position has an obstacle
                try:
                    if obstacle_map[next_r, next_c] == 1:
                        continue
                except IndexError:
                    self.debug(DEBUG_VERBOSE, f"  IndexError for position {next_r, next_c}")
                    continue

                # Check for space-time collisions with other agents
                if (next_r, next_c, next_t) in reservations and reservations[(next_r, next_c, next_t)] != agent:
                    # We can try waiting if the reservation is not for this agent
                    wait_node = (current_r, current_c, next_t)
                    
                    # Ensure waiting position isn't reserved
                    if (current_r, current_c, next_t) not in reservations or reservations[(current_r, current_c, next_t)] == agent:
                        # Compute tentative g_score for waiting
                        tentative_g_score, _ = self._calculate_waiting_cost(current, g_score, came_from, agent)

                        if wait_node not in g_score or tentative_g_score < g_score[wait_node]:
                            # This wait path is better
                            came_from[wait_node] = (current, 6)  # Wait action
                            g_score[wait_node] = tentative_g_score
                            f_score[wait_node] = tentative_g_score + self._heuristic((current_r, current_c), goal)

                            heapq.heappush(open_set, (f_score[wait_node], counter, wait_node))
                            counter += 1

                    # Skip this neighbor (moving to occupied space)
                    continue

                # Check for swap collisions with other agents
                swap_conflict = False
                for other_agent in self.env.agents:
                    if other_agent != agent:
                        # Check if the other agent might be at our current position at next time step
                        if (current_r, current_c, next_t) in reservations and reservations[(current_r, current_c, next_t)] == other_agent:
                            # Check if the other agent is moving to our next position
                            if (next_r, next_c, next_t) in reservations and reservations[(next_r, next_c, next_t)] == other_agent:
                                swap_conflict = True
                                break

                if swap_conflict:
                    # Skip this neighbor (swap collision)
                    continue

                # Compute tentative g_score
                tentative_g_score = g_score[current] + 1  # Cost for moving

                # If this path is better
                if next_node not in g_score or tentative_g_score < g_score[next_node]:
                    # Record this path
                    came_from[next_node] = (current, action)
                    g_score[next_node] = tentative_g_score
                    f_score[next_node] = tentative_g_score + self._heuristic((next_r, next_c), goal)

                    heapq.heappush(open_set, (f_score[next_node], counter, next_node))
                    counter += 1

            # Allow waiting in place as well (but with slightly higher cost)
            wait_node = (current_r, current_c, current_t + 1)

            # Check if waiting position isn't reserved
            if wait_node not in reservations or reservations[wait_node] == agent:
                # Compute tentative g_score for waiting
                tentative_g_score, _ = self._calculate_waiting_cost(current, g_score, came_from, agent)

                if wait_node not in g_score or tentative_g_score < g_score[wait_node]:
                    # This wait path is better
                    came_from[wait_node] = (current, 6)  # Wait action
                    g_score[wait_node] = tentative_g_score
                    f_score[wait_node] = tentative_g_score + self._heuristic((current_r, current_c), goal)
                    
                    heapq.heappush(open_set, (f_score[wait_node], counter, wait_node))
                    counter += 1

        # If we reach here, no path was found
        if explored_nodes >= max_explored:
            self.debug(DEBUG_INFO, f"  Path search exceeded max exploration limit for {agent}")
        return None
    
    def _attempt_path_reconnection(self, agent, current_r, current_c, modified_idx):
        """
        Try to reconnect from the modified position back to the original path.
        
        Args:
            agent: Agent ID
            current_r, current_c: Current position after local adjustment
            modified_idx: Index of the modified step in the path
        
        Returns:
            True if reconnection was successful, False otherwise
        """
        self.debug(DEBUG_INFO, f"Attempting path reconnection for agent {agent}")
        
        # If we're at the end of the path, no need to reconnect
        if modified_idx >= len(self.paths[agent]) - 1:
            return True
        
        # Get the original global path starting from a few steps ahead
        # Try to reconnect to a point 2-4 steps ahead to avoid the obstacle
        max_look_ahead = min(4, len(self.paths[agent]) - modified_idx - 1)
        if max_look_ahead <= 0:
            return False
            
        # Find reconnection candidates from the original path
        reconnect_candidates = []
        for i in range(1, max_look_ahead + 1):
            target_idx = modified_idx + i
            if target_idx < len(self.paths[agent]):
                target_r, target_c, _ = self.paths[agent][target_idx]
                # Calculate Manhattan distance to this point
                distance = abs(current_r - target_r) + abs(current_c - target_c)
                reconnect_candidates.append((target_idx, (target_r, target_c), distance))
        
        if not reconnect_candidates:
            return False
            
        # Sort candidates by distance (closest first)
        reconnect_candidates.sort(key=lambda x: x[2])
        
        # Get obstacle map for planning
        global_state, _ = self.env.get_global_state()
        obstacle_map = global_state[1].copy()
        
        # Try to connect to each candidate
        for target_idx, (target_r, target_c), _ in reconnect_candidates:
            # Plan a mini-path from current position to the reconnection point
            mini_path = self._mini_path_plan(
                (current_r, current_c), 
                (target_r, target_c), 
                obstacle_map
            )
            
            if mini_path:
                # Found a reconnection path!
                self.debug(DEBUG_INFO, f"Found reconnection path of length {len(mini_path)} to original path at index {target_idx}")
                
                # Replace the path segment between modified_idx and target_idx
                new_path = self.paths[agent][:modified_idx]  # Keep path up to modification
                new_path.extend(mini_path)  # Add reconnection segment
                new_path.extend(self.paths[agent][target_idx+1:])  # Add remainder of original path
                
                self.paths[agent] = new_path
                return True
        
        # Could not reconnect to any point
        self.debug(DEBUG_INFO, f"Failed to reconnect path for agent {agent}")
        return False
    
    def _mini_path_plan(self, start, goal, obstacle_map, max_depth=10):
        """
        Simple breadth-first search for short path planning.
        More efficient than A* for very short paths.
        
        Args:
            start: (row, col) Start position
            goal: (row, col) Goal position
            obstacle_map: Binary map where 1 indicates obstacle
            max_depth: Maximum search depth
        
        Returns:
            List of (row, col, action) tuples or None if no path found
        """
        # Use BFS for quick local planning
        queue = deque([(start, [])])  # (position, path)
        visited = {start}
        
        while queue:
            (r, c), path = queue.popleft()
            
            # Check if we've reached the goal
            if (r, c) == goal:
                return path
                
            # Stop if path is too long
            if len(path) >= max_depth:
                continue
                
            # Check all four directions
            for action, (dr, dc) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
                nr, nc = r + dr, c + dc
                
                # Skip if out of bounds
                if nr < 0 or nr >= self.env.grid_size[0] or nc < 0 or nc >= self.env.grid_size[1]:
                    continue
                    
                # Skip if obstacle
                if obstacle_map[nr, nc] == 1:
                    continue
                    
                # Skip if already visited
                if (nr, nc) in visited:
                    continue
                    
                # Add to queue
                new_path = path + [(nr, nc, action)]
                queue.append(((nr, nc), new_path))
                visited.add((nr, nc))
        
        # No path found
        return None
    
    def detect_and_handle_local_conflicts(self, observations):
        """
        Use local observations to detect and handle conflicts with:
        1. Other agents (collisions)
        2. Static obstacles (shelves)
        3. Dynamic obstacles (humans)

        Returns:
            modified (bool): True if any paths were modified
        """

        return False # Disable local conflict handling for now
        modified = False

        for agent, observation in observations.items():
            # Skip if agent has no path or is at the end of its path
            if agent not in self.paths or self.path_indices[agent] >= len(self.paths[agent]):
                continue

            current_pos = self.env.agent_positions[agent]
            next_idx = self.path_indices[agent]
            conflict_detected = False

            # Get next planned action
            if next_idx < len(self.paths[agent]):
                next_r, next_c, action = self.paths[agent][next_idx]

                # We only handle movement actions (0-3)
                if action in [0, 1, 2, 3]:
                    # Calculate expected next position based on current position
                    dr, dc = self.action_to_delta[action]
                    expected_next_r, expected_next_c = current_pos[0] + dr, current_pos[1] + dc
                    
                    # Check if path is out of sync with current position
                    if (expected_next_r, expected_next_c) != (next_r, next_c):
                        self.debug(DEBUG_INFO, f"Agent {agent} path out of sync. Expected: {(expected_next_r, expected_next_c)}, Path: {(next_r, next_c)}")
                        self.need_replanning[agent] = True
                        modified = True
                        continue

                    # Calculate local coordinates - agent is at (2,2) in observation
                    local_r, local_c = 2 + dr, 2 + dc
                    
                    # Check if coordinates are within observation bounds
                    if 0 <= local_r < 5 and 0 <= local_c < 5:
                        # Get obstacle information from observation channels
                        other_agent = observation[1][local_r, local_c] == 1  # Channel 1: other agents
                        static_obstacle = observation[2][local_r, local_c] == 1  # Channel 2: static obstacles
                        dynamic_obstacle = observation[3][local_r, local_c] == 1  # Channel 3: dynamic obstacles
                        
                        # Check actual grid value at intended position
                        grid_value = self.env.grid[next_r, next_c]
                        
                        # Debug output
                        self.debug(DEBUG_VERBOSE, f"Agent {agent} current pos: {current_pos}, local obs: dynamic_obstacle={dynamic_obstacle}, static_obstacle={static_obstacle}, at {local_r},{local_c}, grid {next_r, next_c} value: {grid_value}")
                        
                        # Skip collision detection for pickup/dropoff points that are goals
                        is_pickup = (next_r, next_c) in self.env.pickup_points
                        is_dropoff = (next_r, next_c) in self.env.dropoff_points
                        is_goal = (next_r, next_c) == self.env.agent_goals[agent]
                        
                        # Handle various obstacle scenarios
                        if other_agent or static_obstacle:
                            # Handle non-human obstacles normally
                            if not (is_pickup or is_dropoff):
                                conflict_detected = True
                                self._handle_conflict(agent, observation, action, current_pos, next_idx, "static")
                                modified = True
                        elif dynamic_obstacle:
                            # Special handling for humans
                            conflict_detected = True
                            
                            # Check if this human is directly blocking a goal point
                            is_blocking_goal = False
                            goal_pos = self.env.agent_goals[agent]
                            
                            # Create next_pos tuple from next_r and next_c
                            next_pos = (next_r, next_c)
                            
                            # Check if the human is between agent and goal using Manhattan distance
                            if self._heuristic(current_pos, next_pos) < self._heuristic(current_pos, goal_pos) and \
                               self._heuristic(next_pos, goal_pos) < self._heuristic(current_pos, goal_pos):
                                is_blocking_goal = True
                                
                            # If human is blocking goal, we need to be more aggressive about finding a path around
                            if is_blocking_goal:
                                self.debug(DEBUG_INFO, f"Agent {agent} detected human blocking path to goal")
                                
                                # Try to find a path around the human
                                alternative_action = self._find_path_around_human(agent, observation, action)
                                
                                if alternative_action is not None and alternative_action != 6:
                                    # Apply alternative movement
                                    alt_dr, alt_dc = self.action_to_delta[alternative_action]
                                    alt_r, alt_c = current_pos[0] + alt_dr, current_pos[1] + alt_dc
                                    
                                    # Update path with alternative action
                                    self.paths[agent][next_idx] = (alt_r, alt_c, alternative_action)
                                    self.debug(DEBUG_INFO, f"Agent {agent} rerouted around human with action {alternative_action}")
                                    
                                    # Try to reconnect to original path
                                    reconnection_success = self._attempt_path_reconnection(agent, alt_r, alt_c, next_idx)
                                    
                                    if not reconnection_success:
                                        # If we can't reconnect, force replanning after a short wait
                                        self.debug(DEBUG_INFO, f"Path reconnection failed for agent {agent}, will replan after a few steps")
                                        self.human_avoidance_replanning[agent] = 3  # Replan after 3 steps
                                    
                                    modified = True
                                else:
                                    # If no alternative path found, trigger immediate replanning
                                    self.debug(DEBUG_INFO, f"Agent {agent} cannot find path around human, triggering replanning")
                                    self.need_replanning[agent] = True
                                    modified = True
                            else:
                                # Human is not blocking a goal, handle normally with waiting
                                self._handle_conflict(agent, observation, action, current_pos, next_idx, "dynamic")
                                modified = True
                    
        # Check for agents that need replanning after human avoidance
        if hasattr(self, 'human_avoidance_replanning'):
            for agent in list(self.human_avoidance_replanning.keys()):
                self.human_avoidance_replanning[agent] -= 1
                if self.human_avoidance_replanning[agent] <= 0:
                    self.debug(DEBUG_INFO, f"Agent {agent} triggering delayed replanning after human avoidance")
                    self.need_replanning[agent] = True
                    del self.human_avoidance_replanning[agent]

        return modified

    def _find_path_around_human(self, agent, observation, blocked_action):
        """
        Find a path around a human obstacle by considering multiple steps ahead.
        This is a more thorough search than _find_local_alternative.
        
        Returns: Best alternative action or None if no path found
        """
        current_pos = self.env.agent_positions[agent]
        goal = self.env.agent_goals[agent]
        
        # We'll try to find a 2-step path around the obstacle
        best_score = float('-inf')
        best_action = None
        
        # Try each direction for the first step
        for action1 in range(4):
            if action1 == blocked_action:
                continue
            
            # Get delta for first action
            dr1, dc1 = self.action_to_delta[action1]
            local_r1, local_c1 = 2 + dr1, 2 + dc1
            
            # Check if first step is valid (within bounds and no obstacles)
            if 0 <= local_r1 < 5 and 0 <= local_c1 < 5:
                if (observation[1][local_r1, local_c1] == 0 and  # No agents
                    observation[2][local_r1, local_c1] == 0 and  # No shelves
                    observation[3][local_r1, local_c1] == 0):    # No humans
                    
                    # Calculate position after first step
                    pos1_r = current_pos[0] + dr1
                    pos1_c = current_pos[1] + dc1
                    pos1 = (pos1_r, pos1_c)
                    
                    # Try each direction for the second step
                    for action2 in range(4):
                        dr2, dc2 = self.action_to_delta[action2]
                        pos2_r = pos1_r + dr2
                        pos2_c = pos1_c + dc2
                        pos2 = (pos2_r, pos2_c)
                        
                        # We don't have observation data for pos2, use grid instead
                        if 0 <= pos2_r < self.env.grid.shape[0] and 0 <= pos2_c < self.env.grid.shape[1]:
                            # Check grid value at pos2
                            grid_val = self.env.grid[pos2_r, pos2_c]
                            
                            # If pos2 is empty or a pickup/dropoff point
                            if grid_val == 0 or grid_val == 4 or grid_val == 5:
                                # Calculate how much closer we get to the goal
                                initial_dist = self._heuristic(current_pos, goal)
                                final_dist = self._heuristic(pos2, goal)
                                score = initial_dist - final_dist
                                
                                # Bonus for getting around the human
                                if final_dist < initial_dist:
                                    score += 2.0
                                    
                                # If this is better than our current best, update it
                                if score > best_score:
                                    best_score = score
                                    best_action = action1
        
        # If we found a promising action, return it
        if best_action is not None:
            self.debug(DEBUG_INFO, f"Agent {agent} found path around human with score {best_score}")
            return best_action
            
        # Fall back to simpler alternative finding if no 2-step path works
        return self._find_local_alternative(agent, observation, blocked_action)

    def _handle_conflict(self, agent, observation, action, current_pos, next_idx, obstacle_type):
        """
        Helper method to handle conflict resolution for different obstacle types
        """
        # Find an alternative action
        alternative_action = self._find_local_alternative(agent, observation, action)
        
        if alternative_action is not None and alternative_action != 6:
            # Apply alternative movement
            alt_dr, alt_dc = self.action_to_delta[alternative_action]
            alt_r, alt_c = current_pos[0] + alt_dr, current_pos[1] + alt_dc
            
            # Update path with alternative action
            self.paths[agent][next_idx] = (alt_r, alt_c, alternative_action)
            self.debug(DEBUG_INFO, f"Agent {agent} rerouted around {obstacle_type} obstacle with action {alternative_action}")
            
            # Try to reconnect to original path
            self._attempt_path_reconnection(agent, alt_r, alt_c, next_idx)
            
            return True
        else:
            # No alternative found, must wait
            self.paths[agent].insert(next_idx, (current_pos[0], current_pos[1], 6))
            self.debug(DEBUG_INFO, f"Agent {agent} has no alternative to avoid {obstacle_type} obstacle, waiting")
            
            # Handle waiting counters
            pos_key = (current_pos[0], current_pos[1])
            if not hasattr(self, "position_wait_counts"):
                self.position_wait_counts = {a: {} for a in self.env.agents}
            
            if pos_key not in self.position_wait_counts[agent]:
                self.position_wait_counts[agent][pos_key] = 0
                
            self.position_wait_counts[agent][pos_key] += 1
            
            # Trigger replanning if agent has waited too long in the same position
            if self.position_wait_counts[agent][pos_key] >= self.local_deadlock_threshold:
                self.debug(DEBUG_INFO, f"Agent {agent} waited too long at {pos_key}, triggering replanning")
                self.need_replanning[agent] = True
                self.position_wait_counts[agent][pos_key] = 0
            
            return True

    def _detect_oscillation(self, history, pattern_length):
        """
        Detect oscillation patterns in the position history.
        
        Args:
            history: List of positions
            pattern_length: Length of the oscillation pattern to detect
        Returns:
            True if oscillation detected, False otherwise
        """

        if len(history) < pattern_length * 2:
            return False
        
        # Check if the last n positions match the previous n positions
        for i in range(pattern_length):
            if history[-(i + 1)] != history[-(i + 1 + pattern_length)]:
                return False
            
        return True

    def _find_local_alternative(self, agent, observation, blocked_action):
        """
        Find an alternative action for the agent based on the local observation.
        
        Args:
            agent: Agent ID
            observation: Local observation of the agent
            blocked_action: Action that is blocked
        
        Returns:
            Alternative action or wait if no alternative is found
        """
        # Get agent's current position and goal
        current_pos = self.env.agent_positions[agent]
        goal = self.env.agent_goals[agent]

        # Calculate current manhattan distance to goal
        current_distance = self._heuristic(current_pos, goal)

        # Evaluate all possible directions
        possible_actions = []

        for action in range(4):  # Only movement actions (0-3)
            if action == blocked_action:
                continue  # Skip the blocked action

            # Get the delta for this action
            delta = self.action_to_delta[action]
            dr, dc = delta

            # Convert to local observation coordinates
            local_r = 2 + dr
            local_c = 2 + dc

            # Check if this position is within bounds
            if 0 <= local_r < 5 and 0 <= local_c < 5:
                # Check if this position is free (no obstacles of any kind)
                if (observation[1][local_r, local_c] == 0 and  # No other agents
                    observation[2][local_r, local_c] == 0 and  # No shelves
                    observation[3][local_r, local_c] == 0):    # No humans
                    
                    # Calculate new position and distance to goal
                    new_r = current_pos[0] + dr
                    new_c = current_pos[1] + dc
                    new_pos = (new_r, new_c)
                    
                    # Skip pickup/dropoff points that belong to other agents
                    if ((new_pos in self.env.pickup_points or new_pos in self.env.dropoff_points) and
                        new_pos != self.env.agent_goals[agent]):
                        continue
                    
                    new_distance = self._heuristic(new_pos, goal)

                    # Calculate score: lower distance is better
                    score = current_distance - new_distance
                    
                    # Bonus for actions that move around blocked direction
                    # This encourages moving around obstacles rather than backtracking
                    if (blocked_action in [0, 2] and action in [1, 3]) or \
                       (blocked_action in [1, 3] and action in [0, 2]):
                        score += 0.5  # Small bonus for perpendicular movement
                    
                    possible_actions.append((action, score))

        # If there are possible actions
        if possible_actions:
            # Sort by score (higher is better)
            possible_actions.sort(key=lambda x: x[1], reverse=True)
            
            # Return the best action
            best_action = possible_actions[0][0]
            self.debug(DEBUG_INFO, f"Agent {agent} found alternative action {best_action} with score {possible_actions[0][1]}")
            return best_action

        # If no alternatives found, return wait action
        self.debug(DEBUG_INFO, f"Agent {agent} has no alternatives, waiting")
        return 6  # Wait action

    def handle_post_action_state(self, actions):
        """
        Handle the state after executing pickup / dropoff actions.
        """
        result = False

        for agent, action in actions.items():
            # After executing a dropoff action, force immediate replanning
            if action == 5: # Dropoff action
                self.debug(DEBUG_INFO, f"Agent {agent} executed dropoff action, forcing replanning")
                
                # Reset consecutive waits to prevent false deadlock detection
                self.consecutive_waits[agent] = 0

                # Reset position history to prevent false oscillation detection
                self.position_history[agent] = []
                
                # Reset position-specific wait counts too
                if hasattr(self, "position_wait_counts") and agent in self.position_wait_counts:
                    self.position_wait_counts[agent] = {}

                # Mark this agent for replanning
                self.need_replanning[agent] = True
                result = True

            # After executing a pickup action, force immediate replanning
            elif action == 4: # Pickup action
                self.debug(DEBUG_INFO, f"Agent {agent} executed pickup action, forcing replanning")
                
                # Reset consecutive waits to prevent false deadlock detection
                self.consecutive_waits[agent] = 0

                # Reset position history to prevent false oscillation detection
                self.position_history[agent] = []

                # Mark this agent for replanning
                self.need_replanning[agent] = True
                result = True

        return result

    def _calculate_waiting_cost(self, current, g_score, came_from, agent):
        """
        Calculate cost for waiting based on consecutive wait count.
        """

        # Get the current cost for this node or default to 0
        current_cost = g_score.get(current, 0)

        if current in came_from and came_from[current][1] == 6:
            # Calculate how many times we've been waiting already
            consecutive_waits = 1
            temp_node = current
            while temp_node in came_from and came_from[temp_node][1] == 6:
                consecutive_waits += 1
                temp_node = came_from[temp_node][0]
            
            # Exponential backoff for waiting
            wait_cost = current_cost + 1.0 + (1.5 ** consecutive_waits)

            # Log long waits
            if consecutive_waits > 5:
                self.debug(DEBUG_INFO, f"Agent {agent} has been waiting for {consecutive_waits} steps!")
            return wait_cost, consecutive_waits
        else:
            # Standard wait cost
            return current_cost + 2.0, 0

    def _reconstruct_path(self, came_from, current):
        """
        Reconstruct the path from the came_from map.
        
        Args:
            came_from: Dictionary mapping nodes to their predecessors
            current: Current node to reconstruct path from

        Returns:
            List of (row, col, action) tuples representing the path
        """
        total_path = []
        while current in came_from:
            current_r, current_c, _ = current
            prev_node, action = came_from[current]
            total_path.append((current_r, current_c, action))
            current = prev_node

        # Reverse path (we built it backwards)
        total_path.reverse()
        return total_path

    def _heuristic(self, a, b):
        """
        Manhattan distance heuristic
        
        Args:
            a: (row, col) First position
            b: (row, col) Second position

        Returns:
            Manhattan distance between a and b
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


def run_a_star(env, n_steps=1000, debug_level=DEBUG_INFO):
    """
    Run the A* path planning on the warehouse environment

    Args:
        env: Warehouse environment
        n_steps: Number of steps to run
        debug_level: Debug level (default: INFO)
    """
    # Initialize the A* agent
    a_star_agent = AStarAgent(env)

    # Set debug level
    a_star_agent.debug_level = debug_level

    # Reset the environment
    observations, _ = env.reset()

    # Initial planning for all agents
    a_star_agent.debug(DEBUG_CRITICAL, "\nInitial planning...")
    a_star_agent.plan_all_paths()

    # Cumulative rewards tracking
    cumulative_rewards = {agent: 0 for agent in env.agents}

    # Run the simulation
    for step in range(n_steps):
        # Use local observations to detect and handle conflicts
        local_adjustments = a_star_agent.detect_and_handle_local_conflicts(observations)

        if local_adjustments:
            a_star_agent.debug(DEBUG_INFO, "Local adjustments made to paths")

        # Get actions from A* agent
        actions = a_star_agent.get_actions()
        
        # Step the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Update cumulative rewards
        for agent in env.agents:
            if agent in cumulative_rewards:
                cumulative_rewards[agent] += rewards[agent]

        # Check if any pickup/dropoff actions were performed
        if a_star_agent.handle_post_action_state(actions):
            a_star_agent.debug(DEBUG_INFO, "Replanning after pickup/dropoff...")
            a_star_agent.plan_all_paths()
        
        # Print positions vs goals
        a_star_agent.debug(DEBUG_INFO, f"Step {step} positions:")
        for agent in env.agents:
            pos = env.agent_positions[agent]
            goal = env.agent_goals[agent]
            carrying = env.agent_carrying[agent]
            a_star_agent.debug(DEBUG_INFO, f"{agent}: Pos={pos}, Goal={goal}, Carrying={carrying}")
            
            # Check if agent has reached goal
            if pos == goal:
                if (not carrying and goal in env.pickup_points) or (carrying and goal in env.dropoff_points):
                    a_star_agent.debug(DEBUG_INFO, f"{agent} has reached its goal!")
        
        # Render the environment
        # env.render()
        
        # Print info
        a_star_agent.debug(DEBUG_INFO, f"Rewards: {rewards}")
        a_star_agent.debug(DEBUG_INFO, f"Completed tasks: {env.completed_tasks}")
        
        # Slow down simulation
        #time.sleep(0.3)
        
        # Check for replanning
        replan = any(a_star_agent.need_replanning.values())
        if replan:
            a_star_agent.debug(DEBUG_INFO, "\nReplanning paths...")
            a_star_agent.plan_all_paths()
            
        # Break if all agents are done
        if terminations["__all__"]:
            break

    # Print total delievered tasks
    total_delivered = sum(env.completed_tasks.values())
    a_star_agent.debug(DEBUG_INFO, f"\nTotal delivered tasks: {total_delivered}")
    # Print total scores
    total_score = sum(cumulative_rewards.values())
    a_star_agent.debug(DEBUG_INFO, f"Total scores: {total_score}")
            
    # Close the environment
    env.close()

    # Return total completed tasks
    return total_delivered