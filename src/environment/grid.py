from pettingzoo.utils import wrappers
from pettingzoo.utils import ParallelEnv
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

class WarehouseEnv(ParallelEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'name': 'warehouse_v0'}

    def __init__(self, grid_size=(20, 20), num_agents=2, num_shelves=30, num_dynamic_obstacles=10, 
                 num_pickup_points=3, collision_penalty=-2, task_reward=10, step_cost=-0.1, 
                 render_mode=None):
        super().__init__()

        # Environment parameters
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_shelves = num_shelves
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.num_pickup_points = num_pickup_points
        self.collision_penalty = collision_penalty
        self.task_reward = task_reward
        self.step_cost = step_cost
        self.render_mode = render_mode

        # Agent definitions
        self.agents = ["agent_" + str(i) for i in range(num_agents)]

        # Action spaces: 0: Up, 1: Right, 2: Down, 3: Left, 4: Pickup, 5: Drop, 6: Wait
        self.action_spaces = {agent: spaces.Discrete(7) for agent in self.agents}

        # Observation space components
        # 9 channels: 
        # 1. Agent's position (1 at agent's position, 0 elsewhere)
        # 2. Other agents' positions (1 at other agents' positions, 0 elsewhere)
        # 3. Shelves' positions (1 at shelves' positions, 0 elsewhere)
        # 4. Dynamic obstacles' positions (1 at dynamic obstacles' positions, 0 elsewhere)
        # 5. Current goal position (1 at goal position, 0 elsewhere)
        # 6. Pickup points' positions (1 at pickup points' positions, 0 elsewhere)
        # 7. Dropoff points' positions (1 at dropoff points' positions, 0 elsewhere)
        # 8. Agent's carrying status (1 if agent is carrying, 0 otherwise)
        # 9. Valid pickup/drop indicator (1 if agent is at a valid pickup/drop point, 0 otherwise)

        # Each agent observes a NxN grid around itself, plus some global state information
        self.local_observation_size = (5, 5) # 5x5 grid around the agent
        obs_shape = (9,) + self.local_observation_size
        self.observation_spaces = {agent: spaces.Boz(low=0, high=1, shape=obs_shape, dtype=np.float32) 
                                   for agent in self.agents}
        
        # Initialize grid and agent data
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize grid: 0=empty, 1=agent, 2=shelf, 3=dynamic obstacle, 4=pickup point, 5=dropoff point
        self.grid = np.zeros(self.grid_size, dtype=np.int8)

        # Place shelves in a warehouse-like layout (aisles)
        self._place_shelves()

        # Place pickup points (e.g. packing stations)
        self.pickup_points = self._place_random_points(self.num_pickup_points, [1, 2])
        for pos in self.pickup_points:
            self.grid[pos] = 4

        # Place dropoff points (e.g. shipping area)
        self.delivery_points = self._place_random_points(min(self.num_pickup_points, 3), [1, 2, 4])
        for pos in self.delivery_points:
            self.grid[pos] = 5

        # Initialize agents at random positions
        self.agent_positions = {}
        self.agent_carrying = {agent: False for agent in self.agents}
        self.agent_goals = {}
        self.agent_item_types = {} # What type of item the agent is carrying (possible extension)

        for agent in self.agents:
            # Find empty position
            pos = self._get_random_empty_position()
            self.agent_positions[agent] = pos
            self.grid[pos] = 1

            # Assign initial goal (pickup point)
            self._assign_new_goal(agent)

        # Metrics tracking
        self.steps = 0
        self.completed_tasks = {agent: 0 for agent in self.agents}
        self.collisions = {agent: 0 for agent in self.agents}
        self.total_distance = {agent: 0 for agent in self.agents}

        # Create observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        return observations
    
    def step(self, actions):
        self.steps += 1
        rewards = {agent: self.step_cost for agent in self.agents} # base cost per step
        dones = {agent: False for agent in self.agents}

        # Process agents in a random order to avoid bias
        agents_order = list(self.agents)
        random.shuffle(agents_order)

        # Track which cells wil be occupied after actions
        new_positions = {}

        # First pass: compute new positions
        for agent in agents_order:
            action = actions[agent]
            current_pos = self.agent_positions[agent]
            new_pos = current_pos

            # Execute action
            if action < 4: # Movement actions
                # Compute new position
                delta_row, delta_col = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
                new_row = current_pos[0] + delta_row
                new_col = current_pos[1] + delta_col

                # Check boundaries
                if 0 <= new_row < self.grid_size[0] and 0 <= new_col < self.grid_size[1]:
                    # Check for shelf or dynamic obstacle
                    if self.grid[new_row, new_col] not in [2, 3]:
                        new_pos = (new_row, new_col)
                        self.total_distance[agent] += 1
            
            new_positions[agent] = new_pos

        # Second pass: check for conflicts and update positions
        for agent in agents_order:
            intended_pos = new_positions[agent]

            # Check for collisions with other agents' new positions
            collision = False
            for other_agent, other_pos in new_positions.items():
                if other_agent != agent and other_pos == intended_pos:
                    collision = True
                    break

            if collision:
                # On collision, agent stays in place
                new_positions[agent] = self.agent_positions[agent]
                rewards[agent] += self.collision_penalty
                self.collisions[agent] += 1
            else:
                # Update position if no collision
                self.grid[self.agent_positions[agent]] = 0 # Clear previous position
                self.agent_positions[agent] = intended_pos
                self.grid[intended_pos] = 1 # Mark new position

                # Check if agent reached its goal
                if intended_pos == self.agent_goals[agent]:
                    if not self.agent_carrying[agent]:
                        # Agent has reached pickup point
                        self.agent_carrying[agent] = True
                        rewards[agent] += self.task_reward / 2 # Reward for picking up an item, half of task reward

                        # Remember which item type was picked up
                        pickup_idx = self.pickup_points.index(intended_pos)
                        self.agent_item_types[agent] = pickup_idx % len(self.delivery_points)

                        # Assign new goal (dropoff point)
                        delivery_idx = self.agent_item_types[agent]
                        self.agent_goals[agent] = self.delivery_points[delivery_idx]
                    else:
                        # Agent has reached dropoff point
                        self.agent_carrying[agent] = False
                        rewards[agent] += self.task_reward / 2 # Reward for dropping off an item, half of task reward
                        self.completed_tasks[agent] += 1

                        # Assign new goal (pickup point)
                        self._assign_new_goal(agent)

        # Check for termination criteria (e.g. max steps etc.)
        # In a warehouse setting, we might not have a natural termination point
        # so we'll consider the episode ongoing until explicitly stopped
        dones["__all__"] = False

        # Create observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        # Optional info dict for additional data
        info = {agent: {
            "position:": self.agent_positions[agent],
            "goal": self.agent_goals[agent],
            "carrying": self.agent_carrying[agent],
            "completed_tasks": self.completed_tasks[agent],
        } for agent in self.agents}

        return observations, rewards, dones, info
    
    def render(self):
        """
        Render the warehouse environment
        """

        if self.render_mode is None:
            return
        
        # Create a grid for visualization
        vis_grid = np.zeros(self.grid_size)

        # Mark shelves
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.grid[i, j] == 2: # Shelf
                    vis_grid[i, j] = 2

        # Mark pickup points
        for pos in self.pickup_points:
            vis_grid[pos] = 4 # Pickup point

        # Mark dropoff points
        for pos in self.delivery_points:
            vis_grid[pos] = 5 # Dropoff point

        # Mark agents and their goals
        for i, agent in enumerate(self.agents):
            # Agent position
            pos = self.agent_positions[agent]
            vis_grid[pos] = 5 + i % 5 # Different colors for different agents

            # Agent goal
            goal = self.agent_goals[agent]
            if vis_grid[goal] < 5: # Don't overwrite another agent
                vis_grid[goal] = 10 + i % 5 # Different colors for different agent's goal
        
        # Create color map
        colors = [
            'white',        # 0: Empty
            'grey',         # 1: (not used)
            'black',        # 2: Shelf
            'blue',         # 3: Dynamic obstacle
            'green',        # 4: Pickup point
            'red',          # 5: Dropoff point
            'orange',       # 6: Agent 1
            'purple',       # 7: Agent 2
            'brown',        # 8: Agent 3
            'pink',         # 9: Agent 4
            'cyan',         # 10: Agent 1's goal
            'magenta',      # 11: Agent 2's goal
            'yellow',       # 12: Agent 3's goal
            'lime',         # 13: Agent 4's goal
        ]
        cmap = ListedColormap(colors)

        plt.figure(figsize=(10, 10))
        plt.imshow(vis_grid, cmap=cmap)
        plt.grid(True, color='grey', linestyle='-', linewidth=0.5)
        plt.title(f"Warehouse Environment - Step {self.steps}")

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Shelf'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Pickup Point'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Dropoff Point'),
        ]

        for i, agent in enumerate(self.agents[:4]):
            agent_color = colors[6 + i]
            goal_color = colors[10 + i]
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=agent_color, markersize=10, label=f"Agent {i}"))
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=goal_color, markersize=10, label=f"Agent {i}'s Goal"))

        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        if self.render_mode == 'human':
            plt.pause(0.1)
            plt.show(block=False)
        elif self.render_mode == 'rgb_array':
            fig = plt.gcf()
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close()
            return img
        
    def _get_observation(self, agent):
        """
        Generate observation for an agent
        """

        # Get agent's position
        agent_pos = self.agent_positions[agent]

        # Initialize observation channels
        obs = np.zeros((9,) + self.local_observation_size, dtype=np.float32)

        # Extract the local observation window (5x5 grid around the agent)
        r, c = agent_pos
        window_size = self.local_observation_size[0] // 2

        # For each cell in the observation window
        for i in range(-window_size, window_size + 1):
            for j in range(-window_size, window_size + 1):
                # Calculate global grid coordinates
                gr, gc = r + i, c + j

                # Calculate local observation coordinates
                lr, lc = i + window_size, j + window_size

                # Check if the global coordinates are within bounds
                if 0 <= gr < self.grid_size[0] and 0 <= gc < self.grid_size[1]:
                    # Channel 1: Agent position (only at the center)
                    if i == 0 and j == 0:
                        obs[0, lr, lc] = 1

                    # Channel 2: Other agents' positions
                    cell_has_other_agent = False
                    for other_agent in self.agents:
                        if other_agent != agent and self.agent_positions[other_agent] == (gr, gc):
                            cell_has_other_agent = True
                            break
                    obs[1, lr, lc] = 1 if cell_has_other_agent else 0

                    # Channel 3: Static obstacles (shelves)
                    obs[2, lr, lc] = 1 if self.grid[gr, gc] == 2 else 0

                    # Channel 4: Dynamic obstacles
                    obs[3, lr, lc] = 1 if self.grid[gr, gc] == 3 else 0

                    # Channel 5: Current goal position
                    obs[4, lr, lc] = 1 if (gr, gc) == self.agent_goals[agent] else 0

                    # Channel 6: Pickup points
                    obs[5, lr, lc] = 1 if (gr, gc) in self.pickup_points else 0

                    # Channel 7: Dropoff points
                    obs[6, lr, lc] = 1 if (gr, gc) in self.delivery_points else 0

        # Channel 8: Agent's carrying status
        obs[7, :, :] = 1 if self.agent_carrying[agent] else 0

        # Channel 9: Valid pickup/drop indicator
        if self.agent_carrying[agent]:
            # If carrying, mark valid dropoff points
            delivery_idx = self.agent_item_types[agent]
            for i in range(-window_size, window_size + 1):
                for j in range(-window_size, window_size + 1):
                    gr, gc = r + i, c + j
                    lr, lc = i + window_size, j + window_size
                    if 0 <= gr < self.grid_size[0] and 0 <= gc < self.grid_size[1]:
                        if (gr, gc) == self.delivery_points[delivery_idx]:
                            obs[8, lr, lc] = 1
        else:
            # If not carrying, mark all pickup points as valid
            for i in range(-window_size, window_size + 1):
                for j in range(-window_size, window_size + 1):
                    gr, gc = r + i, c + j
                    lr, lc = i + window_size, j + window_size
                    if 0 <= gr < self.grid_size[0] and 0 <= gc < self.grid_size[1]:
                        if (gr, gc) in self.pickup_points:
                            obs[8, lr, lc] = 1

        return obs

    def _place_shelves(self):
        """
        Place shelves in a warehouse-like layout (aisles)
        """
        
        # Create horizontal shelves with aisles in between
        shelf_width = 3
        aisle_width = 2
        
        for i in range(0, self.grid_size[0], shelf_width + aisle_width):
            if i + shelf_width >= self.grid_size[0]:
                break

            for j in range(self.grid_size[1]):
                for k in range(shelf_width):
                    if j + k < self.grid_size[0]:
                        self.grid[i + k, j] = 2 # place shelf
        
    def _place_random_points(self, num_points, avoid_values):
        """
        Place random points on the grid, avoiding certain cell types
        """
        points = []
        for _ in range(num_points):
            pos = self._get_random_empty_position(avoid_values)
            points.append(pos)
        return points
    
    def _get_random_empty_position(self, avoid_values=[1, 2, 3, 4, 5]):
        """
        Get a random empty position on the grid, avoiding certain cell types
        """
        while True:
            r = random.randint(0, self.grid_size[0] - 1)
            c = random.randint(0, self.grid_size[1] - 1)

            if self.grid[r, c] not in avoid_values:
                return (r, c)
            
    def _assign_new_goal(self, agent):
        """
        Assign a new goal to the agent
        """

        if self.agent_carrying[agent]:
            # If agent is carrying an item, deliver it to appropriate dropoff point
            delivery_idx = self.agent_item_types[agent]
            self.agent_goals[agent] = self.delivery_points[delivery_idx]
        else:
            # If agent is not carrying an item, go to a random pickup point
            pickup_idx = random.randint(0, len(self.pickup_points) - 1)
            self.agent_goals[agent] = self.pickup_points[pickup_idx]

# Wrapper for the environment
def env(grid_size=(20, 20), num_agents=2, num_shelves=30, num_dynamic_obstacles=10, 
        num_pickup_points=3, render_mode=None):
    env = WarehouseEnv(grid_size=grid_size, num_agents=num_agents, num_shelves=num_shelves, 
                       num_dynamic_obstacles=num_dynamic_obstacles, num_pickup_points=num_pickup_points, 
                       render_mode=render_mode)
    env = wrappers.CaptureStdoutWrapper(env)
    return env