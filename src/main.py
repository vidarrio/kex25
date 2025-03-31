import numpy as np
from environment import env
from pettingzoo.test import parallel_api_test

def test_warehouse_env():
    # Create warehouse environment with rendering
    warehouse_env = env(grid_size=(15, 15), n_agents=3, render_mode="human")

    # Reset the environment
    observations = warehouse_env.reset()

    # Test the environment API
    parallel_api_test(warehouse_env, num_cycles=10)

    return

    # Run for some steps
    for _ in range(100):
        # Random actions
        actions = {
            agent: warehouse_env.action_space(agent).sample()
            for agent in warehouse_env.agents
        }

        # Step the environment
        observations, rewards, dones, infos = warehouse_env.step(actions)

        # Render
        warehouse_env.render()

        # Print some info
        print(f"Step: {warehouse_env.steps}")
        print(f"Rewards: {rewards}")
        print(f"Completed tasks: {warehouse_env.completed_tasks}")

        # Stop if all agents are done
        if dones["__all__"]:
            break

    # Close the environment
    warehouse_env.close()

if __name__ == "__main__":
    test_warehouse_env()