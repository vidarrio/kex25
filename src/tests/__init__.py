"""
Test suite for the KEX25 project.

This module contains unit tests for various components:
- test_env.py: Tests for the warehouse environment
- test_a_star.py: Tests for the A* pathfinding algorithm
- test_rl.py: Tests for the reinforcement learning algorithm
"""

# Import test modules
from . import test_env

# Export key test functions if needed
from .test_env import (
    test_api_compliance,
    test_reset,
    test_step,
    test_movement,
    test_pickup_dropoff,
    test_observation_space,
    test_agent_goals,
    test_reward_mechanics,
    test_collision_handling
)

# Define what gets imported with "from tests import *"
__all__ = [
    "test_api_compliance",
    "test_reset",
    "test_step",
    "test_movement",
    "test_pickup_dropoff",
    "test_observation_space",
    "test_agent_goals",
    "test_reward_mechanics",
    "test_collision_handling"
]