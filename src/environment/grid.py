from typing import TypeVar
from pettingzoo.utils import ParallelEnv
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import random 
from collections import deque

from environment.constants import ActionType, CellType, Reward

from .utils import SimpleHumanPlanner, get_random_empty_position, assign_new_human_goal
from matplotlib.widgets import Button

OSCILLATION_THRESHOLD = 4

AgentID = TypeVar("AgentID", bound=str)
ObsType = TypeVar("ObsType", bound=list[list[int]])


def is_within_bounds(position: tuple[int, int], grid_size: tuple[int, int]) -> bool:
    """
    Check if the position is within the grid boundaries
    """
    return (0 <= position[0] < grid_size[0]) and (0 <= position[1] < grid_size[1])

def is_collission(position: tuple[int, int], grid: np.ndarray) -> bool:
    """
    Check if the position collides with an obstacle (shelf or human)
    """
    # print(f"Checking collision at {position} with value {grid[position]}")
    return grid[position] in [CellType.AGENT.value, CellType.SHELF.value, CellType.DYNAMIC_OBSTACLE.value]  # 1=agent, 2=shelf, 3=dynamic obstacle

def move_position(position: tuple[int, int], action: ActionType) -> tuple[int, int]:
    """
    Move the position based on the action
    """
    trans = action.get_transpose()
    return position[0] + trans[0], position[1] + trans[1]
    

class WarehouseEnv(ParallelEnv[str, list[list[int]], int]):
    metadata = {'render_modes': ['human', 'rgb_array'], 'name': 'warehouse_v0'}

    reward_table: Reward = Reward(
        step_penalty=-1.0,
        correct_direction_reward=0.1,
        collision_penalty=-2.0,
        wrong_pickup_penalty=-2.0,
        wrong_dropoff_penalty=-2.0,
        oscillation_penalty = -2.0,
        task_reward=20.0
    )

    def __init__(
            self,
            grid_size=(20, 20),
            human_grid_size=(20, 20),
            n_agents=2,
            n_humans=1,
            num_shelves=30, 
            num_pickup_points=2,
            num_dropoff_points=2,
            observation_size=(5, 5),
            render_mode=None,
            seed=None,
        ):
        
        super().__init__()
        self.seed = seed

        # Size of the whole grid
        self.grid_size = grid_size
        self.observation_size = observation_size
        self.n_agents = n_agents
        self.num_shelves = num_shelves
        self.num_pickup_points = num_pickup_points
        self.num_dropoff_points = num_dropoff_points

        self.possible_agents = ["agent_" + str(i) for i in range(self.n_agents)]
        
        # Size of the human area, with origon at the top left corner (?)
        self.human_grid_size = human_grid_size
        self.n_humans = n_humans
        self.human_planner = SimpleHumanPlanner()
        self.possible_humans = ["human_" + str(i) for i in range(self.n_humans)]
        
        self.render_mode = render_mode

        self.reset(seed=self.seed, options=None)
    
    def reset(self, seed=None, options=None) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Set active agents and humans
        self.agents = self.possible_agents.copy()
        self.humans = self.possible_humans.copy()

        # Initialize grid: 0=empty, 1=agent, 2=shelf, 3=dynamic obstacle, 4=pickup point, 5=dropoff point
        self.grid = np.zeros(self.grid_size, dtype=np.int8)
        self.grid_points = np.zeros(self.grid_size, dtype=np.int8)

        # Place shelves in a warehouse-like layout (aisles)
        self._place_shelves()

        # Place dropoff points (e.g. shipping area)
        self._place_dropoff_points()
        self.pickup_points = []

        # Initialize agents at random positions
        self.agent_positions = {}
        self.agent_prev_positions = {}
        self.agent_carrying = {agent: False for agent in self.agents}
        self.agent_goals = {}
        self.agent_item_types = {agent: None for agent in self.agents}

        self._place_agents_randomly()
        self._assign_pickup_point_to_agents()

        # Initialize humans at random positions
        self.human_positions = {}
        self.human_goals = {}
        
        self._place_humans_randomly()
            
        # Metrics tracking
        self.steps = 0
        self.completed_tasks = {agent: 0 for agent in self.agents}
        self.collisions = {agent: 0 for agent in self.agents}
        self.total_distance = {agent: 0 for agent in self.agents}
        
        # track the last OSCILLATION_THRESHOLD+1 positions of each agent
        self.agent_position_histories = {
            agent: deque(maxlen = OSCILLATION_THRESHOLD + 1)
            for agent in self.agents
        }

        # Create observations
        observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        info = {agent: {} for agent in self.agents}
        return observation_spaces, info

    def _place_agents_randomly(self):
        for agent in self.agents:
            # Find empty position
            pos = get_random_empty_position(grid=self.grid, grid_size=self.grid_size)
            self.agent_positions[agent] = pos
            self.agent_prev_positions[agent] = pos
            self.grid[pos] = CellType.AGENT.value

    def _assign_pickup_point_to_agents(self):
        for agent in self.agents:
            # Assign a new pickup point
            pick_point = self.create_pickup_point()
            self.agent_goals[agent] = pick_point

    def _place_humans_randomly(self):
        for human in self.humans:
            # Find empty position
            pos = get_random_empty_position(grid=self.grid, grid_size=self.human_grid_size)
            self.human_positions[human] = pos
            self.grid[pos] = CellType.DYNAMIC_OBSTACLE.value

            # Assign initial goal (pickup point)
            assign_new_human_goal(human, self.human_goals, self.grid, self.human_grid_size)

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
            self.grid_points[pos] = CellType.DROPOFF_POINT.value
    
    def step(self, actions):
        self.steps += 1

        self._step_humans()
        step_rewards = self._step_agents(actions)

        # Create observations
        observations = {agent: self.observation_space(agent) for agent in self.agents}

        # Optional info dict for additional data
        info = {}
        for agent in self.agents:
            info[agent] = {
                "position": self.agent_positions[agent],
                "goal": self.agent_goals[agent],
                "carrying": self.agent_carrying[agent],
                "completed_tasks": self.completed_tasks[agent],
            }

        return observations, step_rewards, info
    
    def _step_humans(self):
        humans_order = list(self.humans)
        random.shuffle(humans_order)

        # Get actions for humans
        human_actions = self.human_planner.get_actions(
            humans=self.humans,
            human_positions=self.human_positions,
            human_goals=self.human_goals,
            grid=self.grid,
            grid_size=self.human_grid_size
        )

        for human in humans_order:
            human_action = ActionType(human_actions[human])

            if not human_action.is_movement():
                continue
        
            current_pos = self.human_positions[human]
            new_position = move_position(current_pos, human_action)
            if not is_within_bounds(new_position, self.human_grid_size):
                continue
            

            if self.grid[new_position] in [CellType.SHELF.value, CellType.AGENT.value]:
                continue

            self.grid[current_pos] = CellType.EMPTY.value
            self.grid[new_position] = CellType.DYNAMIC_OBSTACLE.value
            self.human_positions[human] = new_position

    def _step_agents(self, actions: dict[AgentID, ActionType]) -> dict[AgentID, float]:
       
        def _handle_move(agent, agent_action) -> float:
            current_pos = self.agent_positions[agent]
            new_pos = move_position(current_pos, agent_action)
            if not is_within_bounds(new_pos, self.grid_size):
                # print("not within bounds")
                return self.reward_table.collision_penalty
            
            if is_collission(new_pos, self.grid):
                # print("collision")
                return self.reward_table.collision_penalty
            
            if current_pos != new_pos and self.grid[new_pos] == CellType.EMPTY.value:
                self.grid[new_pos] = CellType.AGENT.value
                self.grid[current_pos] = CellType.EMPTY.value

            self.agent_prev_positions[agent] = current_pos
            self.agent_positions[agent] = new_pos
            
            reward = self.reward_table.step_penalty
            
            self.agent_position_histories[agent].append(new_pos)

            # 1) punish simple back-and-forth oscillations
            if self._check_oscillation(agent):
                reward += self.reward_table.oscillation_penalty
                # print("--------------------OSCILLATION PENALTY--------------------")

            # 2) if not a 2-point oscillation, punish fixed-length cycles
            elif self._check_cycle(agent):
                reward += self.reward_table.oscillation_penalty
                # print("--------------------CYCLE PENALTY--------------------")

            return reward
        
        def _handle_pickup(agent) -> float:
            # Penalties
            current_pos = self.agent_positions[agent]
            if current_pos not in self.pickup_points:
                return self.reward_table.wrong_pickup_penalty

            if self.agent_carrying[agent]:
                # If we are carrying, we cannot pick up again, penalty
                return self.reward_table.wrong_pickup_penalty

            if current_pos in self.pickup_points and current_pos != self.agent_goals[agent]:
                # At pickup point, but the wrong one
                return self.reward_table.wrong_pickup_penalty
            
            if current_pos == self.agent_goals[agent]:
                # Pickup point matches agent's goal, reward
                self.remove_pickup_point(current_pos)
                self.agent_carrying[agent] = True
                self.grid[current_pos] = CellType.AGENT.value
                self.agent_goals[agent] = self.get_random_dropoff_point()
                return self.reward_table.task_reward
            
            raise ValueError("Error in pickup logic")
            
        def _handle_dropoff(agent) -> float:
            # Penalties
            current_pos = self.agent_positions[agent]
            if not (current_pos in self.dropoff_points):
                return self.reward_table.wrong_dropoff_penalty

            if not self.agent_carrying[agent]:
                # If we are not carrying, we cannot drop off, penalty
                return self.reward_table.wrong_dropoff_penalty
            
            if current_pos in self.dropoff_points and current_pos != self.agent_goals[agent]:
                # At dropoff point, but the wrong one
                return self.reward_table.wrong_dropoff_penalty

            # Rewards
            if current_pos == self.agent_goals[agent]:
                # Dropoff point matches agent's goal, reward
                self.agent_carrying[agent] = False
                self.completed_tasks[agent] += 1
                new_pickup = self.create_pickup_point()
                self.agent_goals[agent] = new_pickup
                return self.reward_table.task_reward
            
            raise ValueError("Error in dropoff logic")
        
        def _handle_wait(agent) -> float:
            # If we are waiting, we get a small penalty
            current_pos = self.agent_positions[agent]
            previous_pos = self.agent_prev_positions[agent]

            if current_pos == previous_pos:
                # If we are not moving, we get a small penalty
                return self.reward_table.step_penalty * 2 * 4
            
            return self.reward_table.step_penalty * 2
        
        def _is_closer(old_pos, new_pos, goal_pos):
            # Check if the new position is closer to the goal
            old_distance = np.linalg.norm(np.array(old_pos) - np.array(goal_pos))
            new_distance = np.linalg.norm(np.array(new_pos) - np.array(goal_pos))
            return new_distance < old_distance
            

        # Process agents and humans in a random order to avoid bias
        agents_order = list(self.agents)
        random.shuffle(agents_order)
        
        rewards = {agent: 0 for agent in agents_order}

        for agent in agents_order:
            agent_action = ActionType(actions[agent])
            
            step_reward = 0
            match agent_action:
                case ActionType.LEFT | ActionType.DOWN | ActionType.RIGHT | ActionType.UP:
                    step_reward = _handle_move(agent, agent_action)
                    prev_pos = self.agent_prev_positions[agent]
                    new_pos = self.agent_positions[agent]
                    goal = self.agent_goals[agent]

                    if _is_closer(prev_pos, new_pos, goal):
                        step_reward += self.reward_table.correct_direction_reward
                case ActionType.PICKUP:
                    step_reward = _handle_pickup(agent)
                case ActionType.DROP:
                    step_reward = _handle_dropoff(agent)
                case ActionType.WAIT:
                    step_reward = _handle_wait(agent)
                case _:
                    step_reward = 0

            rewards[agent] += step_reward
            self.agent_prev_positions[agent] = self.agent_positions[agent]
        return rewards

    def render(self):
        """Render the warehouse environment with support for up to 10 agents"""
        if self.render_mode != 'human':
            return 

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
        self.ax.set_ylim(self.grid_size[0] - 0.5, -0.5)
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

    def observation_space(self, agent: AgentID) -> spaces.Space:
        """
        Generate observation for an agent
        """

        # Get agent's position
        agent_pos = self.agent_positions[agent]

        # Initialize observation channels
        obs = np.zeros((10,) + self.observation_size, dtype=np.float32)

        # Extract the local observation window (5x5 grid around the agent)
        r, c = agent_pos
        window_size = self.observation_size[0] // 2

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

                else:
                    obs[2, lr, lc] = 1
        # Channel 8: Agent's carrying status
        obs[7, :, :] = 1 if self.agent_carrying[agent] else 0

        # Channel 9: Valid pickup/drop indicator
        if self.agent_carrying[agent]:
            # If carrying, mark valid dropoff points
            for i in range(-window_size, window_size + 1):
                for j in range(-window_size, window_size + 1):
                    gr, gc = r + i, c + j
                    lr, lc = i + window_size, j + window_size
                    if 0 <= gr < self.grid_size[0] and 0 <= gc < self.grid_size[1]:
                        if (gr, gc) == self.agent_goals[agent]:
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

        
        # Goal is outside the observation window
        if delta_r < 0 and delta_c == 0:

            obs[9, 0, window_size] = 1
        elif delta_r < 0 and delta_c > 0:

            obs[9, 0, -1] = 1
        elif delta_r == 0 and delta_c > 0:

            obs[9, window_size, -1] = 1
        elif delta_r > 0 and delta_c > 0:

            obs[9, -1, -1] = 1
        elif delta_r > 0 and delta_c == 0:

            obs[9, -1, window_size] = 1
        elif delta_r > 0 and delta_c < 0:

            obs[9, -1, 0] = 1
        elif delta_r == 0 and delta_c < 0:

            obs[9, window_size, 0] = 1
        elif delta_r < 0 and delta_c < 0:

            obs[9, 0, 0] = 1
        
        
        return obs
    
    def action_space(self, agent: AgentID) -> spaces.Space:
        """
        Return the action space for the given agent.
        """
        # The agent can choose among all ActionType values
        return spaces.Discrete(len(ActionType))
    
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
        global_state = np.zeros((9,) + self.grid_size, dtype=np.float32)

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

    def get_pickup_points(self) -> list[tuple[int, int]]:
        """
        Get the pickup points in the environment
        """
        pickup_points = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.grid_points[i, j] == CellType.PICKUP_POINT.value:
                    pickup_points.append((i, j))

        # print(f"Pickup points: {pickup_points}")
        return pickup_points

    def refresh_pickup_points(self):
        self.pickup_points = self.get_pickup_points()

    def get_random_dropoff_point(self) -> tuple[int, int]:
        dropoff_idx = random.randint(0, len(self.dropoff_points) - 1)
        return self.dropoff_points[dropoff_idx]
    
    def create_pickup_point(self):
        """
        Create a new pickup point in the grid
        """
        # Get a random empty position in the grid
        new_pickup_pos = get_random_empty_position(
            grid=self.grid, grid_size=self.grid_size, avoid_values=[
                CellType.AGENT.value,
                CellType.SHELF.value,
                CellType.DYNAMIC_OBSTACLE.value, 
                CellType.PICKUP_POINT.value,
                CellType.DROPOFF_POINT.value
            ]
        )
        
        # Assign the pickup point to the grid
        self.grid_points[new_pickup_pos] = CellType.PICKUP_POINT.value
        self.refresh_pickup_points()
        return new_pickup_pos

    def remove_pickup_point(self, pos):
        """
        Remove a pickup point from the grid
        """
        if pos in self.pickup_points:
            self.grid_points[pos] = CellType.EMPTY.value
            self.refresh_pickup_points()

    def _check_oscillation(self, agent: AgentID) -> bool:
        """
        Return True if the last OSCILLATION_THRESHOLD+1 positions
        of `agent` strictly alternate between two cells.
        """
        history = self.agent_position_histories.get(agent)
        if history is None or len(history) < OSCILLATION_THRESHOLD + 1:
            return False

        seq = list(history)
        # must involve exactly two distinct cells
        if len(set(seq)) != 2:
            return False

        # check that seq[i] == seq[i+2] for all valid i
        for i in range(len(seq) - 2):
            if seq[i] != seq[i+2]:
                return False

        return True
    
    def _check_cycle(self, agent: AgentID) -> bool:
        """
        Detect if the agent has looped through OSCILLATION_THRESHOLD distinct cells
        and returned to the start (i.e. a fixed-length cycle of length OSCILLATION_THRESHOLD).
        """
        history = self.agent_position_histories.get(agent)
        # need at least OSCILLATION_THRESHOLD+1 positions to see a full cycle
        if history is None or len(history) < OSCILLATION_THRESHOLD + 1:
            return False

        seq = list(history)
        # 1) it must start and end at the same cell
        if seq[0] != seq[-1]:
            return False
        # 2) all of the intermediate positions must be distinct
        if len(set(seq[:-1])) != len(seq) - 1:
            return False

        return True
    
#
# Frame stacking wrapper and factory for WarehouseEnv
#

class WarehouseFrameStack(ParallelEnv):
    """
    Wrapper that stacks the last n_frames observations per agent.
    """
    def __init__(self, env: WarehouseEnv):
        super().__init__()
        self.env = env
        self.n_frames = env.n_frames
        # Initialize deques for each agent
        self.frames = {
            agent: deque(maxlen=self.n_frames)
            for agent in self.env.possible_agents
        }

    def reset(self, **kwargs):
        # Reset the base environment
        obs, info = self.env.reset(**kwargs)
        # Fill each deque with the initial observation n_frames times
        for agent, agent_obs in obs.items():
            self.frames[agent].clear()
            for _ in range(self.n_frames):
                self.frames[agent].append(agent_obs.copy())
        # Return stacked observations
        stacked = {
            agent: np.concatenate(list(self.frames[agent]), axis=0)
            for agent in obs
        }
        return stacked, info

    def step(self, actions):
        # Step the base environment
        obs, rewards, info = self.env.step(actions)
        # Append new obs and build stacked observations
        stacked = {}
        for agent, agent_obs in obs.items():
            self.frames[agent].append(agent_obs.copy())
            stacked[agent] = np.concatenate(list(self.frames[agent]), axis=0)
        return stacked, rewards, info

    def observation_space(self, agent) -> spaces.Space:
        """
        Return the stacked observation space: 10 channels per frame times n_frames.
        """
        # Base env produces 10 channels per observation
        H, W = self.env.observation_size
        channels = 10 * self.n_frames
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(channels, H, W),
            dtype=np.float32
        )

    def action_space(self, agent):
        # Pass through the base action_space
        return self.env.action_space(agent)

    def __getattr__(self, name):
        """
        Delegate any missing attributes to the base WarehouseEnv
        """
        return getattr(self.env, name)


def make_env(grid_size=(20,20), human_grid_size=(20,20), n_agents=2, n_humans=1,
             num_shelves=30, num_pickup_points=2, num_dropoff_points=2,
             observation_size=(5,5), render_mode=None, use_frame_stack=True,
             n_frames=8, seed=None):
    """
    Factory to create a WarehouseEnv with optional frame stacking.
    """
    base = WarehouseEnv(
        grid_size=grid_size,
        human_grid_size=human_grid_size,
        n_agents=n_agents,
        n_humans=n_humans,
        num_shelves=num_shelves,
        num_pickup_points=num_pickup_points,
        num_dropoff_points=num_dropoff_points,
        observation_size=observation_size,
        render_mode=render_mode,
        seed=seed
    )
    # Attach frame stack parameters to base env
    base.n_frames = n_frames
    if use_frame_stack:
        return WarehouseFrameStack(base)
    return base