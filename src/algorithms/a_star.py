import heapq
import numpy as np
import time
import random

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

        # Extract obstacle map (1 = obstacle, 0 = free space)
        obstacle_map = global_state[1].copy()

        # Debug print obstacle map shape
        self.debug(DEBUG_VERBOSE, f"Obstacle map shape: {obstacle_map.shape}, Grid size: {self.env.grid_size}")
        self.debug(DEBUG_VERBOSE, f"Transposed shape would be: {obstacle_map.T.shape}")

        # Dictionary to track space-time reservations for collision avoidance
        reservations = {}

        # Debug
        self.debug(DEBUG_CRITICAL, "\n===== Planning Paths =====")

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
                            # Add tiny noise to break symmetry
                            f_score[wait_node] = tentative_g_score + self._heuristic((current_r, current_c), goal) + random.random() * 1e-3

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
                    # Add tiny noise to break symmetry
                    f_score[next_node] = tentative_g_score + self._heuristic((next_r, next_c), goal) + random.random() * 1e-3

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
                    # Add tiny noise to break symmetry
                    f_score[wait_node] = tentative_g_score + self._heuristic((current_r, current_c), goal) + random.random() * 1e-3
                    
                    heapq.heappush(open_set, (f_score[wait_node], counter, wait_node))
                    counter += 1

        # If we reach here, no path was found
        if explored_nodes >= max_explored:
            self.debug(DEBUG_INFO, f"  Path search exceeded max exploration limit for {agent}")
        return None
    
    def detect_and_handle_local_conflicts(self, observations):
        """
        Use local observations to detect and handle:
        1. Dynamic obstacles (humans)
        2. Other agents (collisions)
        3. Static obstacles (shelves)

        Returns:
            modified (boold): True if any paths were modified
        """

        modified = False

        for agent, observation in observations.items():
            # If agent has no path or is at the end of its path, skip
            if agent not in self.paths or self.path_indices[agent] >= len(self.paths[agent]):
                continue

            # Get current position and planned next position
            current_pos = self.env.agent_positions[agent]
            next_idx = self.path_indices[agent]

            # Track if this agent encountered a conflict this iteration
            conflict_detected = False

            # Get the next position and action from the path
            if next_idx < len(self.paths[agent]):
                next_r, next_c, action = self.paths[agent][next_idx]

                # Only check movement actions (0-3)
                if action in [0, 1, 2, 3]:
                    # Convert global planned position to local observation coordinates
                    local_r = next_r - current_pos[0] + 2 # +2 because observations are centered around the agent
                    local_c = next_c - current_pos[1] + 2

                    # Ensure coordinates are within local observation
                    if 0 <= local_r < 5 and 0 <= local_c < 5:
                        
                        # Check all relevant obstacle channels
                        
                        # Channel 1: other agents
                        other_agent = observation[1][local_r, local_c] == 1
                        
                        # Channel 2: static obstacles
                        static_obstacle = observation[2][local_r, local_c] == 1

                        # Channel 3: dynamic obstacles
                        dynamic_obstacle = observation[3][local_r, local_c] == 1

                        if other_agent or static_obstacle or dynamic_obstacle:
                            # Conflict detected
                            conflict_detected = True
                            obstacle_type = "robot" if other_agent else "shelf" if static_obstacle else "human"
                            self.debug(DEBUG_INFO, f"Agent {agent} detected {obstacle_type} at {next_r, next_c}")
                            
                            # If a conflict is detected, modify the path
                            alternative_action = self._find_local_alternative(agent, observation, action)

                            if alternative_action is not None and alternative_action != 6:
                                # Apply the alternative movement action
                                alt_delta = self.action_to_delta[alternative_action]
                                dr, dc = alt_delta
                                alt_r, alt_c = current_pos[0] + dr, current_pos[1] + dc

                                # Update the path with the alternative action
                                self.paths[agent][next_idx] = (alt_r, alt_c, alternative_action)
                                self.debug(DEBUG_INFO, f"Agent {agent} modified path to alternative action {alternative_action}")
                                modified = True
                            else:
                                # If no alternative action is found, wait
                                self.paths[agent].insert(next_idx, (current_pos[0], current_pos[1], 6))  # Wait action
                                self.debug(DEBUG_INFO, f"Agent {agent} has no alternative, waiting")
                                modified = True

                                # Track consecutive waits at this position
                                if not hasattr(self, "position_wait_counts"):
                                    self.position_wait_counts = {a: {} for a in self.env.agents}

                                pos_key = (current_pos[0], current_pos[1])
                                if pos_key not in self.position_wait_counts[agent]:
                                    self.position_wait_counts[agent][pos_key] = 0

                                self.position_wait_counts[agent][pos_key] += 1

                                # If waited too long in the same position, trigger replanning
                                if self.position_wait_counts[agent][pos_key] >= self.local_deadlock_threshold:
                                    self.debug(DEBUG_CRITICAL, f"Agent {agent} has waited too long at {current_pos}, triggering replanning")
                                    self.need_replanning[agent] = True
                                    self.position_wait_counts[agent][pos_key] = 0

                        else:
                            # Clear wait count if no conflict
                            if hasattr(self, "position_wait_counts") and agent in self.position_wait_counts:
                                pos_key = (current_pos[0], current_pos[1])
                                if pos_key in self.position_wait_counts[agent]:
                                    self.position_wait_counts[agent][pos_key] = 0
            

            # If agent had conflict or is already in an oscillation pattern, check for oscillation
            if conflict_detected or len(self.position_history[agent]) >= 2:
                # Record the current position for oscillation detection
                current_pos = self.env.agent_positions[agent]
                self.position_history[agent].append(current_pos)

                # Limit history size
                if len(self.position_history[agent]) > self.oscillation_detection_threshold:
                    self.position_history[agent].pop(0)

                # Check for oscillation patterns
                if len(self.position_history[agent]) >= 4: #  Need at least 4 positions to detect oscillation of length 2
                    # Check for ABAB pattern
                    pattern_length = 2
                    if self._detect_oscillation(self.position_history[agent], pattern_length):
                        self.debug(DEBUG_CRITICAL, f"Agent {agent} detected oscillation pattern, triggering replanning")
                        self.need_replanning[agent] = True
                        self.position_history[agent] = [] # Reset history after detecting oscillation
                        modified = True

        return modified
    
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
                continue # Skip the blocked action

            # Get the delta for this action
            delta = self.action_to_delta[action]
            dr, dc = delta

            # Convert to local observation coordinates
            local_r = 2 + dr
            local_c = 2 + dc

            # Check if this position is within bounds
            if 0 <= local_r < 5 and 0 <= local_c < 5:
                # Check if this position is free (no agents, shelves, or humans)
                if (observation[1][local_r, local_c] == 0 and
                    observation[2][local_r, local_c] == 0 and
                    observation[3][local_r, local_c] == 0):
                    
                    # Calculate new position and distance to goal
                    new_r = current_pos[0] + dr
                    new_c = current_pos[1] + dc
                    new_distance = self._heuristic((new_r, new_c), goal)

                    # Calculate score: lower distance is better, negative score means we're moving away from the goal
                    score = current_distance - new_distance

                    possible_actions.append((action, score))

        # If there are possible actions
        if possible_actions:
            # Sort by score (higher is better)
            possible_actions.sort(key=lambda x: x[1], reverse=True)

            # Return the best action
            return possible_actions[0][0]

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
                self.debug(DEBUG_CRITICAL, f"Agent {agent} executed dropoff action, forcing replanning")
                
                # Reset consecutive waits to prevent false deadlock detection
                self.consecutive_waits[agent] = 0

                # Mark this agent for replanning
                self.need_replanning[agent] = True
                result = True

            # After executing a pickup action, force immediate replanning
            elif action == 4: # Pickup action
                self.debug(DEBUG_CRITICAL, f"Agent {agent} executed pickup action, forcing replanning")
                
                # Reset consecutive waits to prevent false deadlock detection
                self.consecutive_waits[agent] = 0

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
            a_star_agent.debug(DEBUG_CRITICAL, "Replanning after pickup/dropoff...")
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
                    a_star_agent.debug(DEBUG_CRITICAL, f"{agent} has reached its goal!")
        
        # Render the environment
        env.render()
        
        # Print info
        a_star_agent.debug(DEBUG_INFO, f"Rewards: {rewards}")
        a_star_agent.debug(DEBUG_INFO, f"Completed tasks: {env.completed_tasks}")
        
        # Slow down simulation
        # time.sleep(0.1)
        
        # Check for replanning
        replan = any(a_star_agent.need_replanning.values())
        if replan:
            a_star_agent.debug(DEBUG_CRITICAL, "\nReplanning paths...")
            a_star_agent.plan_all_paths()
            
        # Break if all agents are done
        if terminations["__all__"]:
            break
    
    
    
    # Print total delievered tasks
    total_delivered = sum(env.completed_tasks.values())
    a_star_agent.debug(DEBUG_SPECIFIC, f"\nTotal delivered tasks: {total_delivered}")
    # Print total scores
    total_score = sum(cumulative_rewards.values())
    a_star_agent.debug(DEBUG_SPECIFIC, f"Total scores: {total_score}")
            
    # Close the environment
    env.close()
    return total_delivered