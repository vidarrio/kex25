import pytest
import numpy as np
import warnings
from pettingzoo.test import parallel_api_test
import sys
import os

# Add the parent directory to the path to import the env module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import env

@pytest.fixture
def warehouse_env():
    """Create a test environment"""
    with warnings.catch_warnings():
        test_env = env(grid_size=(15, 15), n_agents=3, num_shelves=10, num_dynamic_obstacles=2, num_pickup_points=2, num_dropoff_points=2, render_mode=None)

    yield test_env

def test_api_compliance(warehouse_env):
    """Test the environment complies with the PettingZoo API"""
    with warnings.catch_warnings():
        # We ignore these warnings; they are not relevant for our environment because our agents run infinitely, so should never be terminated or truncated.
        warnings.filterwarnings("ignore", message="Agent was given terminated but was dead last turn")
        warnings.filterwarnings("ignore", message="Agent was given truncated but was dead last turn")
        
        parallel_api_test(warehouse_env)
    
def test_reset(warehouse_env):
    """Test that reset initializes the environment correctly"""
    observations, info = warehouse_env.reset()

    # Check the number of agents
    assert len(warehouse_env.agents) == 3

    # Check that observations are returned for each agent
    assert len(observations) == len(warehouse_env.agents)

    # Check that the info dictionary contains entries for each agent
    assert len(info) == len(warehouse_env.agents)

    # Check that the grid has the right dimensions
    assert warehouse_env.grid.shape == (15, 15)

    # Check that we have the right number of pickup points
    assert len(warehouse_env.pickup_points) == 2

    # Check that we have the right number of dropoff points
    assert len(warehouse_env.dropoff_points) == 2

def test_step(warehouse_env):
    """Test that step executes actions and returns proper results"""
    warehouse_env.reset()

    # Create a dictionary of actions, one per agent
    actions = {agent: 6 for agent in warehouse_env.agents}  # All agents wait

    # Execute the actions
    observations, rewards, terminations, truncations, infos = warehouse_env.step(actions)

    # Check that we get results for each agent
    assert len(observations) == len(warehouse_env.agents)
    assert len(rewards) == len(warehouse_env.agents)

    # Check that terminations and truncations include all possible agents
    # Remove the "__all__" key from the terminations and truncations dictionaries before checking
    terminations_agents = {k: v for k, v in terminations.items() if k != "__all__"}
    truncations_agents = {k: v for k, v in truncations.items() if k != "__all__"}

    assert len(terminations_agents) == len(warehouse_env.possible_agents)
    assert len(truncations_agents) == len(warehouse_env.possible_agents)

    # Check that infos include entries for all agents
    assert len(infos) == len(warehouse_env.possible_agents)

    # Check that rewards for waiting are negative (step cost)
    for reward in rewards.values():
        assert reward < 0

def test_movement(warehouse_env):
    """Test that agents can move in the environment"""
    warehouse_env.reset()

    # Get initial positions of agents
    initial_positions = {agent: warehouse_env.agent_positions[agent] for agent in warehouse_env.agents}

    # Try all movement actions for each agent
    moved = False
    for direction in range(4): # 0: up, 1: right, 2: down, 3: left
        # Create actions dictionary
        actions = {agent: direction for agent in warehouse_env.agents}

        # Execute the actions
        warehouse_env.step(actions)

        # Check that at least one agent has moved
        current_positions = {agent: warehouse_env.agent_positions[agent] for agent in warehouse_env.agents}
        if any(current_positions[agent] != initial_positions[agent] for agent in warehouse_env.agents):
            moved = True
            break

        # In a small grid with obstacles, not all agents may be able to move in every direction
        # so we check that at least one agent has moved
        assert moved, "No agent could move in any direction"

def test_pickup_dropoff(warehouse_env):
    """Test pickup and dropoff mechanics"""
    # This is more of an integration test and depends on the environment state
    # so we will just verify the actions execute without errors
    warehouse_env.reset()

    # Execute pickup action (4) for all agents
    actions = {agent: 4 for agent in warehouse_env.agents}
    warehouse_env.step(actions)

    # Execute dropoff action (5) for all agents
    actions = {agent: 5 for agent in warehouse_env.agents}
    warehouse_env.step(actions)

    # Check that the environment is still functioning
    assert len(warehouse_env.agents) == 3

def test_observation_space(warehouse_env):
    """Test that observations match the defined space"""
    observations, _ = warehouse_env.reset()

    for agent, obs in observations.items():
        # Check observation shape
        expected_shape = (9, 5, 5) # 9 channels, 5x5 grid
        assert obs.shape == expected_shape

        # Check observation bounds
        assert np.all(obs >= 0)
        assert np.all(obs <= 1)

        # Check observation type
        assert obs.dtype == np.float32

def test_agent_goals(warehouse_env):
    """Test that the agents are assignted proper goals"""
    warehouse_env.reset()

    # Each agent should have a goal
    for agent in warehouse_env.agents:
        assert agent in warehouse_env.agent_goals

        # Goal should be a tuple / list of 2 integers (position coordinates)
        goal = warehouse_env.agent_goals[agent]
        assert isinstance(goal, (tuple))
        assert len(goal) == 2
        assert all(isinstance(x, (int, np.integer)) for x in goal)

        # Goal should be either a pickup point (if not carrying anything) or a dropoff point (if carrying something)
        if warehouse_env.agent_carrying[agent]:
            assert goal in warehouse_env.dropoff_points
        else:
            assert goal in warehouse_env.pickup_points

def test_reward_mechanics(warehouse_env):
    """Test the reward mechanics"""
    # This is more complex to test thoroughly, but we can check basic reward values
    warehouse_env.reset()

    # Execute wait action, which should incur a step cost
    actions = {agent: 6 for agent in warehouse_env.agents}
    _, rewards, _, _, _ = warehouse_env.step(actions)

    # Each agent should recieve the step cost
    for agent in warehouse_env.agents:
        assert rewards[agent] == warehouse_env.step_cost

def test_collision_handling(warehouse_env):
    """Test that collisions are handled correctly"""
    warehouse_env.reset()

    # Place two agents next to each other for testing collisions
    agents = list(warehouse_env.agents)
    if len(agents) >= 2:
        # Find an empty position
        for i in range(warehouse_env.grid_size[0]):
            for j in range(warehouse_env.grid_size[1]):
                if warehouse_env.grid[i, j] == 0: # Empty cell
                    # Place first agent here
                    warehouse_env.grid[warehouse_env.agent_positions[agents[0]]] = 0 # Clear old position
                    warehouse_env.agent_positions[agents[0]] = (i, j)
                    warehouse_env.grid[i, j] = 1 # Set new position

                    # Place second agent in adjacent cell if possible
                    if i+1 < warehouse_env.grid_size[0] and warehouse_env.grid[i+1, j] == 0:
                        warehouse_env.grid[warehouse_env.agent_positions[agents[1]]] = 0 # Clear old position
                        warehouse_env.agent_positions[agents[1]] = (i+1, j)
                        warehouse_env.grid[i+1, j] = 1 # Set new position

                        # Try to make the agents collide
                        action = {
                            agents[0]: 1,  # Move right
                            agents[1]: 3,  # Move left
                            agents[2]: 6,  # Wait
                        }

                        # Store initial positions to check if they change
                        initial_position_0 = warehouse_env.agent_positions[agents[0]]
                        initial_position_1 = warehouse_env.agent_positions[agents[1]]

                        # Execute the actions
                        _, rewards, terminations, truncations, infos = warehouse_env.step(action)

                        # Check that agents didn't move (collision)
                        assert warehouse_env.agent_positions[agents[0]] == initial_position_0
                        assert warehouse_env.agent_positions[agents[1]] == initial_position_1

                        # Check for collision penalty
                        assert rewards[agents[0]] <= warehouse_env.collision_penalty
                        assert rewards[agents[1]] <= warehouse_env.collision_penalty
                        return
    # If we reach here, the test failed
    assert False, "Collision test failed: agents did not collide as expected"
