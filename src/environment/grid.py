from pettingzoo.utils import wrappers
from pettingzoo.utils import ParallelEnv
import numpy as np
import gymnasium
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import functools
import os
import sys
from .utils import SimpleHumanPlanner, get_random_empty_position, assign_new_human_goal
from matplotlib.widgets import Button

class WarehouseEnv(ParallelEnv):
    metadata = {'render_modes': ['human', 'rgb_array'], 'name': 'warehouse_v0'}

    def __init__(self, grid_size=(20, 20), human_grid_size=(20, 20), n_agents=2, n_humans=1, num_shelves=30, 
                 num_pickup_points=3, num_dropoff_points=2, collision_penalty=-2, task_reward=100, step_cost=-2,
                 observation_size=(5, 5), render_mode=None):
        super().__init__()

        # Environment parameters
        self.grid_size = grid_size
        self.human_grid_size = human_grid_size
        self.observation_size = observation_size
        self.n_agents = n_agents
        self.n_humans = n_humans
        self.num_shelves = num_shelves
        self.num_pickup_points = num_pickup_points
        self.num_dropoff_points = num_dropoff_points
        self.collision_penalty = collision_penalty
        self.task_reward = task_reward
        self.step_cost = step_cost
        self.render_mode = render_mode

        # Create list of possible agents
        self.possible_agents = ["agent_" + str(i) for i in range(self.n_agents)]

        # Hold every agents previous position
        self.previous_positions = {agent: None for agent in self.possible_agents}
        
        # Create list of possible humans
        self.possible_humans = ["human_" + str(i) for i in range(self.n_humans)]
        
        # Create a SimpleHumanPlanner to handle human actions
        self.human_planner = SimpleHumanPlanner()

        # Initialize grid and agent data
        self.reset()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Return the action space for a specific agent
        The action space is discrete with the following actions:
        0: Move Left (decrease column)
        1: Move Down (increase row)
        2: Move Right (increase column)
        3: Move Up (decrease row)
        4: Pickup Item
        5: Drop Item
        6: Wait
        """
        return spaces.Discrete(7)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Return the observation space for a specific agent
        The observation is a 9-channel grid of size (5, 5) around the agent
        with the following channels:
        1. Agent's position (1 at agent's position, 0 elsewhere)
        2. Other agents' positions (1 at other agents' positions, 0 elsewhere)
        3. Shelves' positions (1 at shelves' positions, 0 elsewhere)
        4. Dynamic obstacles' positions (1 at dynamic obstacles' positions, 0 elsewhere)
        5. Current goal position (1 at goal position, 0 elsewhere)
        6. Pickup points' positions (1 at pickup points' positions, 0 elsewhere)
        7. Dropoff points' positions (1 at dropoff points' positions, 0 elsewhere)
        8. Agent's carrying status (1 if agent is carrying, 0 otherwise)
        9. Valid pickup/drop indicator (1 if agent is at a valid pickup/drop point, 0 otherwise)
        """

        obs_shape = (10,) + self.local_observation_size
        return spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Set active agents
        self.agents = self.possible_agents.copy()
        
        # Set active humans
        self.humans = self.possible_humans.copy()

        # Initialize local observation size
        self.local_observation_size = (5, 5) # 5x5 grid around the agent
        
        # Initialize grid: 0=empty, 1=agent, 2=shelf, 3=dynamic obstacle, 4=pickup point, 5=dropoff point
        self.grid = np.zeros(self.grid_size, dtype=np.int8)

        # Place shelves in a warehouse-like layout (aisles)
        self._place_shelves()

        # Place pickup points (e.g. packing stations)
        self.pickup_points = self._place_random_points(self.num_pickup_points, [1, 2])
        for pos in self.pickup_points:
            self.grid[pos] = 4
            
            
        # Place dropoff points (e.g. shipping area)
        # self.dropoff_points = self._place_random_points(self.num_dropoff_points, [1, 2, 4])
        # for pos in self.dropoff_points:
        #     self.grid[pos] = 5
        # Place dropoff points (e.g. shipping area)
        self._place_dropoff_points()

        # Initialize agents at random positions
        self.agent_positions = {}
        self.agent_carrying = {agent: False for agent in self.agents}
        self.agent_goals = {}
        self.agent_item_types = {agent: None for agent in self.agents}

        for agent in self.agents:
            # Find empty position
            pos = get_random_empty_position(grid=self.grid, grid_size=self.grid_size)
            self.agent_positions[agent] = pos
            self.grid[pos] = 1

            # Assign initial goal (pickup point)
            self._assign_new_goal(agent)
            
         # Initialize humans at random positions
        self.human_positions = {}
        self.human_goals = {}

        for human in self.humans:
            # Find empty position
            pos = get_random_empty_position(grid=self.grid, grid_size=self.human_grid_size)
            self.human_positions[human] = pos
            self.grid[pos] = 3

            # Assign initial goal (pickup point)
            assign_new_human_goal(human, self.human_goals, self.grid, self.human_grid_size)
            
        # Metrics tracking
        self.steps = 0
        self.completed_tasks = {agent: 0 for agent in self.agents}
        self.collisions = {agent: 0 for agent in self.agents}
        self.total_distance = {agent: 0 for agent in self.agents}

        # Create observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        # Create info dict for additional data
        info = {agent: {
            "position": self.agent_positions[agent],
            "goal": self.agent_goals[agent],
            "carrying": self.agent_carrying[agent],
        } for agent in self.agents}

        return observations, info
    
    def step(self, actions):
        self.steps += 1

        # Initialize for active agents
        rewards = {agent: self.step_cost for agent in self.agents} # base cost per step

        # Initialize for all possible agents
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        
        # Initialize for all possible humans
        human_terminations = {human: False for human in self.possible_humans}
        human_truncations = {human: False for human in self.possible_humans}

        # Mark agents not in active list as terminated
        for agent in self.possible_agents:
            if agent not in self.agents:
                terminations[agent] = True
                
        # Mark humans not in active list as terminated
        for human in self.possible_humans:
            if human not in self.humans:
                human_terminations[human] = True

        # Process agents and humans in a random order to avoid bias
        agents_order = list(self.agents)
        humans_order = list(self.humans)
        random.shuffle(agents_order)
        random.shuffle(humans_order)

        # Process humans first

        # Get actions for humans
        human_actions = self.human_planner.get_actions(humans=self.humans, human_positions=self.human_positions,
                                                    human_goals=self.human_goals, grid=self.grid, grid_size=self.human_grid_size)
        # First pass: compute new positions for humans
        humans_new_positions = {}
        for human in humans_order:
            human_action = human_actions[human]
            current_pos = self.human_positions[human]
            new_pos = current_pos # Default: stay in place

            # Execute movement actions (0-3)
            if human_action < 4: # Movement actions
                # Compute new position
                delta_row, delta_col = [(0, -1), (1, 0), (0, 1), (-1, 0)][human_action]
                new_row = current_pos[0] + delta_row
                new_col = current_pos[1] + delta_col

                # Check boundaries and obstacles
                if (0 <= new_row < self.grid_size[0] and 
                    0 <= new_col < self.grid_size[1] and
                    self.grid[new_row, new_col] not in [1, 2]): # Not an agent or shelf
                        new_pos = (new_row, new_col)
            
            # Store intended position
            humans_new_positions[human] = new_pos

        # Second pass: Resolve conflicts

        # Current positions of all agents and humans
        current_positions = {agent: self.agent_positions[agent] for agent in self.agents}
        human_current_positions = {human: self.human_positions[human] for human in self.humans}

        human_reserved_positions = {}

        # Resolve human conflicts
        final_human_positions, human_collisions, human_reserved_positions = second_pass(
            humans_order,
            human_current_positions,
            humans_new_positions,
            human_actions,
            self.humans,
            human_reserved_positions,
            allow_overlap=True
        )

        # Third pass: Update all human positions
        for human in self.humans:
            current_pos = self.human_positions[human]
            new_pos = final_human_positions[human]

            if current_pos != new_pos:
                # Clear old position in grid
                self.grid[current_pos] = 0
                # Update human position
                self.human_positions[human] = new_pos
                # Mark new position in grid
                self.grid[new_pos] = 3  # Human

        # Process agents

        # First pass: compute new positions
        new_positions = {}
        for agent in agents_order:
            action = actions[agent]
            current_pos = self.agent_positions[agent]
            new_pos = current_pos # Default: stay in place

            # Execute movement actions (0-3)
            if action < 4: # Movement actions
                # Compute new position
                delta_row, delta_col = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]  # Left, Down, Right, Up
                new_row = current_pos[0] + delta_row
                new_col = current_pos[1] + delta_col

                # Check boundaries and obstacles
                if (0 <= new_row < self.grid_size[0] and 
                    0 <= new_col < self.grid_size[1] and
                    self.grid[new_row, new_col] not in [2, 3]): # Not a shelf or human
                        new_pos = (new_row, new_col)
                        self.total_distance[agent] += 0.3
            
            # Store intended position
            new_positions[agent] = new_pos
        
        
        
        # Second pass: Resolve conflicts
        reserved_positions = {}
        # Resolve conflicts
        final_positions, collision_agents, reserved_positions = second_pass(
            agents_order, 
            current_positions, 
            new_positions, 
            actions, 
            self.agents,
            reserved_positions,
            allow_overlap=False
        )
        
        # Third pass: Update all positions
        for agent in self.agents:
            current_pos = self.agent_positions[agent]
            new_pos = final_positions[agent]

            if current_pos != new_pos:
                # Clear old position in grid
                self.grid[current_pos] = 0
                # Update agent position
                self.agent_positions[agent] = new_pos
                # Mark new position in grid
                self.grid[new_pos] = 1

            # Apply collision penalties
            if agent in collision_agents:
                rewards[agent] += self.collision_penalty
                self.collisions[agent] += 1

        # Fourth pass: Handle pickup/dropoff/wait actions
        for agent in self.agents:
            action = actions[agent]
            current_pos = self.agent_positions[agent]
            self._fourth_pass(agent, action, current_pos, rewards)

        # Check for termination criteria (e.g. max steps etc.)
        # In a warehouse setting, we might not have a natural termination point
        # so we'll consider the episode ongoing until explicitly stopped
        # Terminations are natural endings (e.g. all agents have completed their tasks)
        # Truncations are artificial endings (e.g. max steps reached)
        terminations["__all__"] = False
        truncations["__all__"] = False

        # Update previous positions for all agents
        for agent in self.agents:
            self.previous_positions[agent] = self.agent_positions[agent]

        # Create observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        # Optional info dict for additional data
        info = {}
        for agent in self.possible_agents:
            if agent in self.agents:
                info[agent] = {
                    "position": self.agent_positions[agent],
                    "goal": self.agent_goals[agent],
                    "carrying": self.agent_carrying[agent],
                    "completed_tasks": self.completed_tasks[agent],
                }
            else:
                # Basic info for inactive agents
                info[agent] = {"active": False}

        # Verify no agents are sharing positions
        position_count = {}
        for agent, pos in self.agent_positions.items():
            if pos not in position_count:
                position_count[pos] = []
            position_count[pos].append(agent)

        for pos, agents in position_count.items():
            if len(agents) > 1:
                print(f"WARNING: Position {pos} is shared by agents: {agents}")
        
        return observations, rewards, terminations, truncations, info
    
    def render(self):
        """Render the warehouse environment with support for up to 10 agents"""
        if self.render_mode == 'human':
            # Initialize the figure on first call
            if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
                self.fig = plt.figure(figsize=(12, 10), num="Warehouse Environment")
                self.ax = plt.subplot(111)
                
                # Create a more attractive button instead of a checkbox
                self.button_ax = plt.axes([0.05, 0.02, 0.15, 0.05])  # [left, bottom, width, height]
                self.show_goal_lines = True  # Default state
                self.button_text = "Hide Goal Lines" if self.show_goal_lines else "Show Goal Lines"
                self.button = Button(
                    self.button_ax, self.button_text,
                    color='lightblue', hovercolor='skyblue'
                )
                self.button.on_clicked(self._toggle_goal_lines)
                
                # Store agent goal lines for toggling visibility
                self.goal_lines = []
            else:
                # Clear the previous content but keep the figure
                self.ax.clear()
                self.goal_lines = []  # Clear old goal lines references

            # Draw a white background grid first
            background = np.zeros(self.grid_size)
            self.ax.imshow(background, cmap='Greys', alpha=0.1, origin='upper')

            # Set up the grid
            self.ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
            self.ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
            self.ax.set_xticks(np.arange(-0.5, self.grid_size[1], 1), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.grid_size[0], 1), minor=True)
            self.ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)

            # Remove axis labels and ticks for cleaner look
            self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            # Get figure size to calculate dynamic font and table scaling
            fig_width, fig_height = self.fig.get_size_inches()
            scale_factor = min(fig_width/12.0, fig_height/10.0)  # Base scale on original size

            # Define title and info text font sizes that scale with window size
            title_size = max(14, int(16 * scale_factor))
            info_size = max(10, int(12 * scale_factor))
            info_text = f"Step: {self.steps}   |   Agents: {len(self.agents)}   |   Humans: {len(self.humans)}"
            
            # Create title with main title and info text
            title = f"Multi-Agent Warehouse Environment\n\n{info_text}"
            
            # Set title with proper padding
            self.ax.set_title(title, 
                             fontsize=title_size,
                             fontweight='bold', 
                             pad=25)
            
            # Position title and adjust styling
            title_obj = self.ax.title
            title_obj.set_y(1.05)
            
            try:
                # Split title into separate components for better styling
                title_text = title_obj.get_text()
                lines = title_text.split('\n')
                
                # Clear existing title
                self.ax.set_title("")
                
                # Add main title at the top
                self.ax.text(0.5, 1.08, lines[0], 
                            transform=self.ax.transAxes,
                            ha='center', va='center', 
                            fontsize=title_size, 
                            fontweight='bold')
                
                # Add info text with background below title
                if len(lines) > 2:
                    self.ax.text(0.5, 1.03, lines[2],
                                transform=self.ax.transAxes,
                                ha='center', va='center', 
                                fontsize=info_size,
                                fontweight='bold',
                                bbox=dict(facecolor='lightblue', alpha=0.5, boxstyle='round,pad=0.5'))
            except Exception:
                # Fallback if text splitting fails
                pass
            
            # Create space for title and info
            self.fig.subplots_adjust(top=0.85)

            # Draw shelves as black squares
            shelf_r, shelf_c = [], []
            for r in range(self.grid_size[0]):
                for c in range(self.grid_size[1]):
                    if self.grid[r, c] == 2:  # Shelf
                        shelf_r.append(r)
                        shelf_c.append(c)
            if shelf_r:
                self.ax.scatter(shelf_c, shelf_r, s=180, marker='s', color='black', label='Shelf')

            # Draw pickup points as green grid cells
            for pos in self.pickup_points:
                r, c = pos
                rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, facecolor='limegreen', alpha=0.4, edgecolor='limegreen')
                self.ax.add_patch(rect)

            # Draw dropoff points as red grid cells
            for pos in self.dropoff_points:
                r, c = pos
                rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, facecolor='tomato', alpha=0.4, edgecolor='tomato')
                self.ax.add_patch(rect)

            # Define distinct colors for up to 10 agents
            agent_colors = [
                'darkorange', 'mediumblue', 'purple', 'deeppink', 'teal',
                'darkgoldenrod', 'darkred', 'forestgreen', 'brown', 'slateblue'
            ]

            # Create legend elements for environment components
            env_legend_elements = [
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=12, label='Shelf'),
                plt.Rectangle((0, 0), 1, 1, facecolor='limegreen', alpha=0.4, label='Pickup Point'),
                plt.Rectangle((0, 0), 1, 1, facecolor='tomato', alpha=0.4, label='Dropoff Point'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=12, label='Human')
            ]

            # Prepare agent legend elements
            agent_legend_elements = []

            # Draw each agent with its goal
            for i, agent in enumerate(self.agents):
                color_idx = i % len(agent_colors)
                agent_color = agent_colors[color_idx]
                pos = self.agent_positions[agent]
                goal = self.agent_goals[agent]
                carrying = self.agent_carrying[agent]

                # Draw agent as a colored circle with black edge
                self.ax.scatter(pos[1], pos[0], s=180, marker='o',
                                color=agent_color,
                                edgecolors='black', linewidth=1.5)
                
                # Add + or - to indicate carrying status
                if carrying:
                    self.ax.text(pos[1], pos[0] - 0.02, "+", 
                                ha='center', va='center', fontsize=14, 
                                fontweight='bold', color='white')
                else:
                    self.ax.text(pos[1], pos[0], "-", 
                                ha='center', va='center', fontsize=16, 
                                fontweight='bold', color='white')

                # Connect agent to goal with a dotted line
                line = self.ax.plot([pos[1], goal[1]], [pos[0], goal[0]], 
                            color=agent_color, linestyle='--', linewidth=2.0, alpha=0.8)[0]
                self.goal_lines.append(line)
                
                # Set line visibility based on toggle state
                line.set_visible(self.show_goal_lines)
                
                # Add agent to legend with carrying status
                carry_status = "+" if carrying else "-"
                agent_legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=agent_color,
                            markersize=12,
                            label=f"Agent {i} ({carry_status})")
                )

            # Draw humans as black circles
            for human in self.humans:
                human_color = "black"
                pos = self.human_positions[human]
                
                self.ax.scatter(pos[1], pos[0], s=180, marker='o',
                                color=human_color,
                                edgecolors='black', linewidth=1.5)
            
            # Add environment legend
            first_legend = self.ax.legend(handles=env_legend_elements,
                                        bbox_to_anchor=(1.05, 1),
                                        loc='upper left',
                                        title="Environment")

            # Keep first legend when adding second
            self.ax.add_artist(first_legend)
            
            # Add agents legend
            second_legend = self.ax.legend(handles=agent_legend_elements,
                                        bbox_to_anchor=(1.25, 1),
                                        loc='upper left',
                                        title="Agents")

            # Calculate dynamic font sizes for tables
            table_font_size = max(8, int(10 * scale_factor))
            title_font_size = max(10, int(12 * scale_factor))
            
            # Prepare data for completed deliveries table
            table_data = []
            for i, agent in enumerate(sorted(self.agents)):
                color_idx = i % len(agent_colors)
                agent_color = agent_colors[color_idx]
                table_data.append([f"Agent {i}", f"{self.completed_tasks[agent]}"])

            # Create main table with delivery counts
            table_height = min(0.3, 0.05 * len(table_data) + 0.1)
            table = self.ax.table(cellText=table_data,
                                colLabels=["Agent", "Deliveries"],
                                loc='center right',
                                cellLoc='center',
                                bbox=[1.05, 0.25, 0.2, table_height])

            # Style main table with dynamic sizing
            table.auto_set_font_size(False)
            table.set_fontsize(table_font_size)
            table.scale(1.2 * scale_factor, 1.5 * scale_factor)

            # Style table cells with colors
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_text_props(fontweight='bold')
                    cell.set_facecolor('lightgray')
                elif j == 0:  # Agent column
                    agent_idx = i - 1
                    color_idx = agent_idx % len(agent_colors)
                    color = plt.matplotlib.colors.to_rgba(agent_colors[color_idx], alpha=0.2)
                    cell.set_facecolor(color)
                elif j == 1:  # Deliveries column
                    cell.set_facecolor('#f8f8f8')

            # Calculate table width based on content
            title_text = "Completed Deliveries"
            table_width = max(0.2, len(title_text) * 0.015 * scale_factor)  
            
            # Create title table for deliveries section
            title_bbox = [1.05, 0.55, table_width, 0.05]
            title_table = self.ax.table(cellText=[[title_text]],
                                        loc='center right',
                                        cellLoc='center',
                                        bbox=title_bbox)

            # Style title table
            title_cell = title_table._cells[(0, 0)]
            title_cell.set_text_props(fontweight='bold', fontsize=title_font_size)
            title_cell.set_facecolor('lightgray')
            title_table.auto_set_font_size(False)
            title_table.set_fontsize(title_font_size)
            title_table.scale(1.2 * scale_factor, 1.0 * scale_factor)
            
            # Ensure both tables use consistent width
            table._bbox = [1.05, 0.25, table_width, table_height]

            # Draw figure and display
            self.fig.tight_layout()
            self.fig.canvas.draw()
            plt.pause(0.01)
            return self.fig

    def _toggle_goal_lines(self, event):
        """Toggle visibility of goal lines when button is clicked"""
        self.show_goal_lines = not self.show_goal_lines
        
        # Update the button text based on current state
        self.button_text = "Hide Goal Lines" if self.show_goal_lines else "Show Goal Lines"
        self.button.label.set_text(self.button_text)
        
        # Toggle the visibility of the lines
        for line in self.goal_lines:
            line.set_visible(self.show_goal_lines)
        
        self.fig.canvas.draw_idle()  # Redraw the figure to show changes

    def _get_observation(self, agent):
        """
        Generate observation for an agent
        """

        # Get agent's position
        agent_pos = self.agent_positions[agent]

        # Initialize observation channels
        obs = np.zeros((10,) + self.local_observation_size, dtype=np.float32)

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
                    obs[6, lr, lc] = 1 if (gr, gc) in self.dropoff_points else 0

        # Channel 8: Agent's carrying status
        obs[7, :, :] = 1 if self.agent_carrying[agent] else 0

        # Channel 9: Valid pickup/drop indicator
        if self.agent_carrying[agent]:
            # If carrying, mark valid dropoff points
            dropoff_idx = self.agent_item_types[agent]
            for i in range(-window_size, window_size + 1):
                for j in range(-window_size, window_size + 1):
                    gr, gc = r + i, c + j
                    lr, lc = i + window_size, j + window_size
                    if 0 <= gr < self.grid_size[0] and 0 <= gc < self.grid_size[1]:
                        if (gr, gc) == self.dropoff_points[dropoff_idx]:
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
                            
        # Channel 10: Compass-like direction to the agent's goal
        goal_pos = self.agent_goals[agent]
        goal_r, goal_c = goal_pos
        delta_r, delta_c = goal_r - r, goal_c - c

        if abs(delta_r) > window_size or abs(delta_c) > window_size:
            # Goal is outside the observation window
            if delta_r < 0 and delta_c == 0:
                # North
                obs[9, 0, window_size] = 1
            elif delta_r < 0 and delta_c > 0:
                # Northeast
                obs[9, 0, -1] = 1
            elif delta_r == 0 and delta_c > 0:
                # East
                obs[9, window_size, -1] = 1
            elif delta_r > 0 and delta_c > 0:
                # Southeast
                obs[9, -1, -1] = 1
            elif delta_r > 0 and delta_c == 0:
                # South
                obs[9, -1, window_size] = 1
            elif delta_r > 0 and delta_c < 0:
                # Southwest
                obs[9, -1, 0] = 1
            elif delta_r == 0 and delta_c < 0:
                # West
                obs[9, window_size, 0] = 1
            elif delta_r < 0 and delta_c < 0:
                # Northwest
                obs[9, 0, 0] = 1
        else:
            # Goal is within the observation window
            direction_r = delta_r + window_size
            direction_c = delta_c + window_size
            obs[9, direction_r, direction_c] = 1
        return obs
    
    def _get_human_observation(self, human):
        """
        Generate observation for an human
        """

        # Get human's position
        human_pos = self.human_positions[human]

        # Initialize observation channels
        obs = np.zeros((9,) + self.local_observation_size, dtype=np.float32)

        # Extract the local observation window (5x5 grid around the human)
        r, c = human_pos
        window_size = self.local_observation_size[0] // 2

        # For each cell in the observation window
        for i in range(-window_size, window_size + 1):
            for j in range(-window_size, window_size + 1):
                # Calculate global grid coordinates
                gr, gc = r + i, c + j

                # Calculate local observation coordinates
                lr, lc = i + window_size, j + window_size

                # Check if the global coordinates are within bounds
                if 0 <= gr < self.human_grid_size[0] and 0 <= gc < self.human_grid_size[1]:
                    # Channel 1: human position (only at the center)
                    if i == 0 and j == 0:
                        obs[0, lr, lc] = 1

                    # Channel 2: Other humans' positions
                    cell_has_other_human = False
                    for other_human in self.humans:
                        if other_human != human and self.human_positions[other_human] == (gr, gc):
                            cell_has_other_human = True
                            break
                    obs[1, lr, lc] = 1 if cell_has_other_human else 0

                    # Channel 3: Static obstacles (shelves)
                    obs[2, lr, lc] = 1 if self.grid[gr, gc] == 2 else 0

                    # Channel 5: Current goal position
                    obs[4, lr, lc] = 1 if (gr, gc) == self.human_goals[human] else 0


        return obs
    
    def get_global_state(self):
        """
        Get the global state of the environment (for CTDE)
        """

        # Create a full grid representation with multiple channels
        global_state = np.zeros((6,) + self.grid_size, dtype=np.float32)

        # Channel 0: All agents' positions (combines 'agent position' and 'other agents' positions')
        for agent, pos in self.agent_positions.items():
            global_state[0, pos[0], pos[1]] = 1

        # Channel 1: Shelves (static obstacles)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.grid[i, j] == 2:
                    global_state[1, i, j] = 1

        # Channel 2: Dynamic obstacles
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.grid[i, j] == 3:
                    global_state[2, i, j] = 1

        # Channel 3: Pickup points
        for pos in self.pickup_points:
            global_state[3, pos[0], pos[1]] = 1

        # Channel 4: Dropoff points
        for pos in self.dropoff_points:
            global_state[4, pos[0], pos[1]] = 1

        # Channel 5: Agent carrying status
        for agent, carrying in self.agent_carrying.items():
            if carrying:
                pos = self.agent_positions[agent]
                global_state[5, pos[0], pos[1]] = 1

        # Additional info as a dictionary
        additional_info = {
            "agent_goals": {agent: goal for agent, goal in self.agent_goals.items()},
            "agent_carrying": self.agent_carrying,
            "agent_item_types": self.agent_item_types,
            "steps": self.steps,
            "completed_tasks": self.completed_tasks
        }

        return global_state, additional_info

    def _place_shelves(self):
        """
        Place shelves in a warehouse-like layout with proper aisles
        - Shelf blocks are 2 wide, 4 long
        - Vertical aisles are 2 cells wide
        - Perimeter has 2-cell wide clearance
        - Shelf count respects self.num_shelves
        """
        # First clear grid
        self.grid[:, :] = 0
        
        # Parameters for shelf layout
        shelf_width = 2       # Width of each shelf block (vertical size)
        shelf_length = 4      # Length of each shelf block (horizontal size)
        aisle_width = 2       # Width of vertical aisles between shelf blocks
        perimeter_width = 2   # Empty space around the perimeter
        
        # Calculate usable area (excluding perimeter)
        usable_height = self.grid_size[0] - 2 * perimeter_width
        usable_width = self.grid_size[1] - 2 * perimeter_width
        
        # Calculate spacing
        h_spacing = shelf_width + aisle_width    # Vertical spacing between shelf blocks
        w_spacing = shelf_length + aisle_width   # Horizontal spacing between shelf blocks
        
        # Calculate maximum number of shelf blocks
        num_row_blocks = (usable_height + aisle_width) // h_spacing  # +aisle_width because we don't need an aisle after the last row
        num_col_blocks = (usable_width + aisle_width) // w_spacing   # +aisle_width because we don't need an aisle after the last column
        
        # Calculate total number of shelf cells available
        total_shelf_cells = num_row_blocks * num_col_blocks * shelf_width * shelf_length
        
        # Limit by specified number
        shelf_target = min(self.num_shelves, total_shelf_cells)
        shelf_cells_placed = 0
        
        # Place shelves
        for row_block in range(num_row_blocks):
            if shelf_cells_placed >= shelf_target:
                break
                
            for col_block in range(num_col_blocks):
                if shelf_cells_placed >= shelf_target:
                    break
                    
                # Calculate the top-left corner of this shelf block
                start_row = perimeter_width + row_block * h_spacing
                start_col = perimeter_width + col_block * w_spacing
                
                # Place the shelf cells for this block
                for i in range(shelf_width):
                    if start_row + i >= self.grid_size[0]:
                        continue
                        
                    for j in range(shelf_length):
                        if start_col + j >= self.grid_size[1]:
                            continue
                            
                        if shelf_cells_placed < shelf_target:
                            self.grid[start_row + i, start_col + j] = 2  # Mark as shelf
                            shelf_cells_placed += 1
        
        # Ensure all aisles are clear
        # Vertical aisles
        for col_block in range(num_col_blocks + 1):
            aisle_col = perimeter_width + col_block * w_spacing - aisle_width
            if col_block > 0:  # Only clear aisles between shelf blocks
                for r in range(perimeter_width, self.grid_size[0] - perimeter_width):
                    for c in range(aisle_width):
                        if 0 <= aisle_col + c < self.grid_size[1]:
                            self.grid[r, aisle_col + c] = 0  # Clear aisle
        
        # Horizontal aisles
        for row_block in range(num_row_blocks + 1):
            aisle_row = perimeter_width + row_block * h_spacing - aisle_width
            if row_block > 0:  # Only clear aisles between shelf blocks
                for c in range(perimeter_width, self.grid_size[1] - perimeter_width):
                    for r in range(aisle_width):
                        if 0 <= aisle_row + r < self.grid_size[0]:
                            self.grid[aisle_row + r, c] = 0  # Clear aisle
        
        # Ensure perimeter is clear
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                if r < perimeter_width or r >= self.grid_size[0] - perimeter_width or \
                   c < perimeter_width or c >= self.grid_size[1] - perimeter_width:
                    self.grid[r, c] = 0  # Clear perimeter

    def _place_random_points(self, num_points, avoid_values):
        """
        Place random points on the grid, avoiding certain cell types
        """
        points = []
        for _ in range(num_points):
            pos = get_random_empty_position(grid=self.grid, grid_size=self.grid_size, avoid_values=avoid_values)
            points.append(pos)
        return points


    def _place_dropoff_points(self):
        """
        Hardcode dropoff points in the corners of the grid.
        """
        # Corners of the grid
        corners = [
            (0, 0),  # Top-left corner
            (0, self.grid_size[1] - 1),  # Top-right corner
            (self.grid_size[0] - 1, 0),  # Bottom-left corner
            (self.grid_size[0] - 1, self.grid_size[1] - 1)  # Bottom-right corner
        ]
        # Middle point of the grid
        # Middle points of the edges
        middle_edges = [
            (0, self.grid_size[1] // 2),  # Middle of the bottom edge
            (self.grid_size[0] - 1, self.grid_size[1] // 2),  # Middle of the top edge
            (self.grid_size[0] // 2, 0),  # Middle of the left edge
            (self.grid_size[0] // 2, self.grid_size[1] - 1)  # Middle of the right edge
        ]
        all_dropoff_points = corners + middle_edges
        
        self.dropoff_points = all_dropoff_points[:self.num_dropoff_points]
        for pos in self.dropoff_points:
            self.grid[pos] = 5
            
    def _assign_new_goal(self, agent):
        """
        Assign a new goal to the agent
        """
        
        if self.agent_carrying[agent]:
            # If agent is carrying an item, deliver it to appropriate dropoff point
            # dropoff_idx = self.agent_item_types[agent] ------Not sure about why there is item_typs here-------
            # dropoff_idx should be based on the dropoff points
            dropoff_idx = random.randint(0, len(self.dropoff_points) - 1)
            
            self.agent_goals[agent] = self.dropoff_points[dropoff_idx]
        else:
            # If agent is not carrying an item, go to a random pickup point
            pickup_idx = random.randint(0, len(self.pickup_points) - 1)
            self.agent_goals[agent] = self.pickup_points[pickup_idx]
            
    def _fourth_pass(self, agent, action, current_pos, rewards):

        # Distance to goal
        goal = self.agent_goals[agent]
        prev_pos = self.previous_positions[agent]

        # Reward for moving closer to goal
        if prev_pos != None:
            prev_dist = abs(prev_pos[0] - goal[0]) + abs(prev_pos[1] - goal[1])
            curr_dist = abs(current_pos[0] - goal[0]) + abs(current_pos[1] - goal[1])
            if curr_dist < prev_dist:
                rewards[agent] += 1

        # Handle pickup action (4)
        if action == 4:
            # Check if agent is at a pickup point and not carrying an item
            if current_pos in self.pickup_points and not self.agent_carrying[agent]:
                # Check if this pickup point matches the agent's goal
                if current_pos == self.agent_goals[agent]:
                    # Agent can pick up item
                    self.agent_carrying[agent] = True
                    rewards[agent] += self.task_reward  # Reward for picking up an item

                    # Remember which item type was picked up
                    pickup_idx = self.pickup_points.index(current_pos)
                    self.agent_item_types[agent] = pickup_idx % len(self.dropoff_points)

                    # Assign new goal (dropoff point)
                    self._assign_new_goal(agent)

                else:
                    # Wrong pickup point, small penalty
                    rewards[agent] += -1

        # Handle dropoff action (5)
        elif action == 5:
            # Check if agent is at a dropoff point and carrying an item
            if current_pos in self.dropoff_points and self.agent_carrying[agent]:
                # Check if this dropoff point matches the agent's goal
                if current_pos == self.agent_goals[agent]:
                    # Agent can drop off item
                    self.agent_carrying[agent] = False
                    rewards[agent] += self.task_reward  # Reward for dropping off an item

                    # Increment completed tasks
                    self.completed_tasks[agent] += 1

                    # Assign new goal (pickup point)
                    self._assign_new_goal(agent)
                else:
                    # Wrong dropoff point, small penalty
                    rewards[agent] += -1

        # Handle wait action (6)
        elif action == 6:
            # No action taken, just wait
            # Add a small penalty for waiting
            factor = 1
            if prev_pos == current_pos:
                factor = 4
            rewards[agent] += 2 * self.step_cost * factor
            pass

def second_pass(entities_order, current_positions, new_positions, actions, all_entities, reserved_positions, allow_overlap=False):
    final_positions = {}
    collision_entities = set()
    
    # Initial claims on positions:
    for entity in all_entities:
        pos = current_positions[entity]
        if pos not in reserved_positions:
            reserved_positions[pos] = entity

    for entity in entities_order:
        action_val = actions[entity]
        intended_pos = new_positions[entity]
        current_pos = current_positions[entity]
        
        # Non-movement actions (4-6): stay in place.
        if action_val >= 4:
            if current_pos in reserved_positions and reserved_positions[current_pos] != entity:
                collision_entities.add(entity)
            else:
                final_positions[entity] = current_pos
                reserved_positions[current_pos] = entity
            continue
        
        # For movement actions (0-3):
        if intended_pos in reserved_positions:
            occupant = reserved_positions[intended_pos]
            # Allow overlap only if:
            #   - allow_overlap flag is True,
            #   - and both entity and occupant are humans.
            if allow_overlap and entity.startswith("human") and occupant.startswith("human"):
                final_positions[entity] = intended_pos
            else:
                collision_entities.add(entity)
                # Stay in current position.
                if current_pos not in reserved_positions or reserved_positions[current_pos] == entity:
                    final_positions[entity] = current_pos
                    reserved_positions[current_pos] = entity
                else:
                    final_positions[entity] = current_pos
            continue
        
        # Check for swapping positions:
        swap_collision = any(
            other_entity != entity and
            current_positions[other_entity] == intended_pos and
            new_positions.get(other_entity) == current_pos and
            not allow_overlap
            for other_entity in all_entities
        )
        if swap_collision:
            final_positions[entity] = current_pos
            collision_entities.add(entity)
            reserved_positions[current_pos] = entity
        else:
            final_positions[entity] = intended_pos
            reserved_positions[intended_pos] = entity

    # Ensure every entity has a final position.
    for entity in all_entities:
        if entity not in final_positions:
            final_positions[entity] = current_positions[entity]
    
    return final_positions, collision_entities, reserved_positions

# Wrapper for the environment
def env(grid_size=(20, 20), human_grid_size=(20, 20), n_agents=2, n_humans=1, num_shelves=30, 
        num_pickup_points=3, num_dropoff_points=2, render_mode=None):
    env = WarehouseEnv(grid_size=grid_size, n_agents=n_agents, n_humans=n_humans, human_grid_size=human_grid_size, num_shelves=num_shelves, 
                        num_pickup_points=num_pickup_points, num_dropoff_points=num_dropoff_points, render_mode=render_mode)
    # env = wrappers.CaptureStdoutWrapper(env)
    return env