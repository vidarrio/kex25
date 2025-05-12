from dataclasses import dataclass
from algorithms import run_a_star, run_q_learning, train_DQN
import os

from environment.grid import WarehouseEnv, make_env
from algorithms.PPO import train_PPO, evaluate_PPO

# PPO curriculum training function
def train_PPO_curriculum(max_steps=1000):
    """
    Train PPO across a sequence of increasingly difficult environments.
    """
    stages = [
        (
            "stage1_simple",
            dict(
                grid_size=(10, 10), human_grid_size=(10, 10),
                n_agents=1, n_humans=0, num_shelves=0,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            1700
        ),
        (
            "stage2_shelves8",
            dict(
                grid_size=(10, 8), human_grid_size=(10, 8),
                n_agents=1, n_humans=0, num_shelves=8,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            6000
        ),
        
        (
            "stage3_shelves16",
            dict(
                grid_size=(10, 8), human_grid_size=(10, 8),
                n_agents=1, n_humans=0, num_shelves=16,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            6000
        ),
        (
            "stage4_shelves32",
            dict(
                grid_size=(21, 20), human_grid_size=(20, 20),
                n_agents=1, n_humans=0, num_shelves=128,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            6000
        ),
        (
            "stage5_shelves64",
            dict(
                grid_size=(34, 32), human_grid_size=(10, 10),
                n_agents=1, n_humans=0, num_shelves=2048,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            6000
        ),
        (
            "stage6_humans",
            dict(
                grid_size=(10, 10), human_grid_size=(10, 10),
                n_agents=1, n_humans=4, num_shelves=0,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            5000
        ),
        (
            "stage7_robots",
            dict(
                grid_size=(10, 10), human_grid_size=(10, 10),
                n_agents=4, n_humans=0, num_shelves=0,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            3000
        ),
        (
            "stage8_full",
            dict(
                grid_size=(21, 20), human_grid_size=(10, 10),
                n_agents=1, n_humans=3, num_shelves=64,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            7000
        ),
        (
            "stage9_advanced",
            dict(
                grid_size=(34, 32), human_grid_size=(34, 32),
                n_agents=5, n_humans=10, num_shelves=2048,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            10000
        ),
    ]

    for name, env_kwargs, episodes in stages:
        print(f"\n=== PPO Curriculum Stage: {name} ({episodes} episodes) ===")
        env_instance = WarehouseEnv(**env_kwargs)
        train_PPO(env_instance, n_episodes=episodes, max_steps=max_steps, save_prefix=name)
    print("\n=== PPO Curriculum Complete ===")



def ppo_eval():
    # Build the same env you trained on
        warehouse_env = WarehouseEnv(
            grid_size=(10, 10), human_grid_size=(10, 10),
            n_agents=1, n_humans=0, num_shelves=0,
            num_pickup_points=1, num_dropoff_points=8,
            render_mode="human"
        )
        # Point this to the .pth you saved after training
        model_path = get_path("stage1_simple")
        evaluate_PPO(
            warehouse_env,
            model_path,
            n_episodes=5,
            max_steps=1000,
            render=True
        )

# Construct path to the model directory
def get_path(name) -> str | None:
    if name is None:
        return None
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    model_name = name + ".pth"
    model_path = os.path.join(models_dir, model_name)
    return model_path
    


def eval_DQN():
    # Create warehouse environment with rendering
    warehouse_env = make_env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=5, n_humans=10, num_shelves=2048, num_pickup_points=3,
                        num_dropoff_points=8, render_mode="human", use_frame_stack=True, n_frames=8)

    # Run DQN agent
    run_q_learning(warehouse_env, n_steps=1000, model_path=get_path("phase_3_dqn_20250511-232139_best"))


@dataclass
class Curriculum:
    grid_size: tuple
    n_agents: int
    n_humans: int
    num_shelves: int
    num_pickup_points: int
    num_dropoff_points: int

    n_episodes: int
    max_steps: int
    save_every: int = 100

    @property
    def name(self):
        return f"{self.grid_size[0]}x{self.grid_size[1]}_{self.n_agents}robot_{self.n_humans}humans_{self.num_pickup_points}pickup_{self.num_dropoff_points}dropoff_{self.num_shelves}shelves_{self.n_episodes}episodes_{self.max_steps}steps(prioritized replay)"



def train_DQL():
    curriculums = [
        Curriculum(grid_size=(34, 32), n_agents=10, n_humans=10, num_shelves=2048, num_pickup_points=1, num_dropoff_points=8, n_episodes=30000, max_steps=1000, save_every=10),
        
    ]

    previous_model: str | None = None
    # previous_model = "34x32_10robot_10humans_4pickup_8dropoff_2048shelves_30001episodes_1000steps"

    for current_curriculum in curriculums:
        environment = make_env(
            grid_size=current_curriculum.grid_size,
            human_grid_size=current_curriculum.grid_size,
            n_agents=current_curriculum.n_agents,
            n_humans=current_curriculum.n_humans,
            num_shelves=current_curriculum.num_shelves,
            num_pickup_points=current_curriculum.num_pickup_points,
            num_dropoff_points=current_curriculum.num_dropoff_points,
            render_mode="human",
            use_frame_stack=True,
            n_frames=8
        )
        train_DQN(
            environment,
            n_episodes=current_curriculum.n_episodes,
            max_steps=current_curriculum.max_steps,
            save_every=current_curriculum.save_every,
            model_path=get_path(current_curriculum.name),
            load_path=get_path(previous_model)
        )
        
        previous_model = current_curriculum.name
        print(f"Trained curriculum: {current_curriculum.name}")


def test_a_star():
    # Create warehouse environment with rendering
    simplest_env = WarehouseEnv(grid_size=(20, 20), n_agents=3, n_humans=4, num_shelves=5, num_pickup_points=4,
                        num_dropoff_points=4, render_mode="human")
    warehouse_env = WarehouseEnv(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=10, n_humans=10, num_shelves=2048, num_pickup_points=4,
                        num_dropoff_points=8, render_mode="human")

    # run_a_star(simplest_env, n_steps=1000, debug_level=5)
    run_a_star(warehouse_env, n_steps=1000, debug_level=5)

# Take first argument as task ("a_star" or "dqn_train" or "dqn_eval")
def main(task):
    if task == "a_star":
        test_a_star()
    elif task == "dqn_train":
        train_DQL()
    elif task == "dqn_eval":
        eval_DQN()
    elif task == "ppo_train_curriculum":
        train_PPO_curriculum()
    elif task == "ppo_eval":
        ppo_eval()
    else:
        raise ValueError("Invalid task. Choose 'a_star', 'dqn_train', 'dqn_eval', 'ppo_train_curriculum', or 'ppo_eval'.")
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <task>")
        print("Tasks: a_star, dqn_train, dqn_eval")
        sys.exit(1)
    
    task = sys.argv[1]
    main(task)