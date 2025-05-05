import os
import sys
import warnings
import numpy
import pytest
import random

# Add the parent directory to the path to import the env module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import env

@pytest.fixture
def stack_frame_test_envs():
    """Create 3 test environments"""
    with warnings.catch_warnings():
        no_frame_env = env(grid_size=(15, 15), human_grid_size=(15, 15), n_agents=1, num_shelves=10, n_humans=0, num_pickup_points=1, num_dropoff_points=1, render_mode=None, use_frame_stack=False, n_frames=1)

        single_frame_env = env(grid_size=(15, 15), human_grid_size=(15, 15), n_agents=1, num_shelves=10, n_humans=0, num_pickup_points=1, num_dropoff_points=1, render_mode=None, use_frame_stack=True, n_frames=1)

        quad_frame_env = env(grid_size=(15, 15), human_grid_size=(15, 15), n_agents=1, num_shelves=10, n_humans=0, num_pickup_points=1, num_dropoff_points=1, render_mode=None, use_frame_stack=True, n_frames=4)

    yield (no_frame_env, single_frame_env, quad_frame_env)

def test_compare_single_frame_and_base_envs(stack_frame_test_envs):
    """Compare the single frame and base environments"""

    # Unpack the environments
    (no_frame_env, single_frame_env, _) = stack_frame_test_envs

    # Random seed
    seed = random.randint(0, 10000)

    # Reset the environments (same seed)
    (obs_no_frame, _) = no_frame_env.reset(seed=seed)
    (obs_single_frame, _) = single_frame_env.reset(seed=seed)

    # Compare the observation shapes and values
    numpy.testing.assert_equal(obs_no_frame, obs_single_frame, "Observations are not equal after reset")

    # Take a step in both environments
    action = no_frame_env.action_space('agent_0').sample()
    actions = {'agent_0': action}

    (obs_no_frame, reward_no_frame, _, _, _) = no_frame_env.step(actions)
    (obs_single_frame, reward_single_frame, _, _, _) = single_frame_env.step(actions)

    # Compare the observation shapes and values
    numpy.testing.assert_equal(obs_no_frame, obs_single_frame, "Observations are not equal after step")
    # Compare the rewards
    numpy.testing.assert_equal(reward_no_frame, reward_single_frame, "Rewards are not equal after step")




