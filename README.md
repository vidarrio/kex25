# kex25

## Overview
This repository contains the code for our bachelor thesis at KTH, which focuses on robot planning in a warehouse environment. The project involves setting up a simulated grid environment, implementing an A* algorithm, and implementing a reinforcement learning algorithm. The goal is to compare the efficiency and performance of these two algorithms.

## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Running the A* Algorithm](#running-the-a-algorithm)
  - [Training the Reinforcement Learning Model](#training-the-reinforcement-learning-model)
  - [Running the Reinforcement Learning Model](#running-the-reinforcement-learning-model)
  - [Running Tests](#running-tests)
- [Algorithms](#algorithms)
  - [A* Algorithm](#a-algorithm)
  - [Reinforcement Learning Algorithm](#reinforcement-learning-algorithm)
- [Dependencies](#dependencies)
- [Contact](#contact)
- [License](#license)

## Project Structure
The project is structured as follows:
```
Project Root/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── src
    ├── algorithms
    │   ├── __init__.py
    │   ├── a_star.py
    │   └── rl.py
    ├── environment
    │   ├── __init__.py
    │   ├── grid.py
    │   └── utils.py
    ├── tests
    │   ├── __init__.py
    │   ├── test_a_star.py
    │   └── test_rl.py
    └── main.py
```

## Setup
The code is compatible with python 3.13.2. After cloning the repository, we also recommend using a virtual environment to manage the dependencies. To set up the virtual environment, run the following commands in the root directory of the project:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Running the A* Algorithm
To run the A* algorithm, run the following command:
```bash

```

### Training the Reinforcement Learning Model
To train the reinforcement learning model, run the following command:
```bash

```

### Running the Reinforcement Learning Model
To run the reinforcement learning model, run the following command:
```bash

```

### Running tests
To run all tests:
```bash
pytest src/tests/
```

To run tests for a specific component:
```bash
pytest src/tests/test_<component>.py
```
where `<component>` is the name of the component you want to test (e.g., `a_star` or `rl`).

To run a specific test function:
```bash
pytest src/tests/test_<component>.py::test_<function>
```
where `<function>` is the name of the test function you want to run (e.g., `test_api_compliance` or `test_collision_handling`).

To run tests with verbose output, use the `-v` flag:
```bash
pytest -v src/tests/
```

To run tests with print statements, use the `-s` flag:
```bash
pytest -s src/tests/
```

## Algorithms

### A* Algorithm
The A* algorithm is a pathfinding algorithm that is used to find the shortest path between two points. The algorithm uses a heuristic function to estimate the cost of reaching the goal from a given point. The algorithm is guaranteed to find the shortest path if the heuristic function is admissible, i.e., it never overestimates the cost of reaching the goal.

It is implemented in the [a_star.py](src/algorithms/a_star.py) file.

### Core algorithm structure

**1. Centralized global planning**

* 3D reservation system (x, y, time) prevents collisions
* Priority based planning (carrying agents first, then non-goal agents)
* Manhattan distance heuristic for path cost estimation

**2. Decentralized execution**

* Local observation-based conflict resolution
* Reactive path adjustments during execution
* Local alternative finding when obstacles appear

### Path planning components

**1. Space-time reservations**

* 3D reservation system (x, y, time) prevents collisions
* Maximum reservation horizon of 15 time steps
* Final positions reserved for all future time steps

**2. Path Cost Calculation**

* Movement cost: 1 per time step
* Wait cost: 2.0 (higher to encourage movement)
* Exponential backoff for consecutive waits: current_cost + 1 + 1.5<sup>(wait_count)</sup>

**3. Priority plannign**

* Agents carrying items get highest priority
* Agents not at goals get medium priority
* Agents at goals get lowest priority

### Conflict Detection and Resolution
**1. Local Observation Window**

* 5x5 grid centered on each agent
* Detects three types of obstacles:
  * Other agents (channel 1)
  * Static obstacles / shelves (channel 2)
  * Dynamic obstacles / humans (channel 3)

**2. Alternative Path Finding**

* Evaluates all movement actions except blocked ones
* Scores alternatives based on distance to goal
* Chooses the best alternative based on the score (lower is better)

**3. Position-Specific Wait Tracking**

* Tracks consecutive waits at specific positions
* Triggers replanning if wait count exceeds local deadlock threshold (default: 3)

### Deadlock Detection
**1. Position-Based Deadlock Detection**

* Tracks waits at specific positions
* Triggers replanning if wait count exceeds local deadlock threshold (default: 3)

**2. Global Deadlock Detection**

* Tracks consecutive wait actions regardless of position
* Triggers replanning if wait count exceeds global deadlock threshold (default: 5)

**3. Oscilation Detection**

* Tracks position history up to oscillation detection threshold (default: 6)
* Identifies repetetive position patterns (ABAB)
* Detects minimum pattern length of 2

### Reinforcement Learning Algorithm
The reinforcement learning algorithm is a model-free machine learning algorithm that learns to make decisions by interacting with the environment. The algorithm learns a policy that maps states to actions by maximizing the expected cumulative reward. The algorithm uses a neural network to approximate the Q-function, which estimates the expected cumulative reward of taking an action in a given state.

It is implemented in the [rl.py](src/algorithms/rl.py) file.

## Dependencies
The project uses the following python dependencies:
- numpy==2.2.3
- scipy==1.15.2
- matplotlib==3.10.1
- torch==2.6.0
- pytest==8.3.5
- gymnasium==1.1.0

## Contact
For any questions or feedback, please contact us at
- Lucas Kristiansson: [luckri@hotmail.se](mailto:luckri@hotmail.se)
- Felix Winkelmann: [felix.winkelmann02@gmail.com](mailto:felix.winkelmann02@gmail.com)

## License
This project is licenced under GPL-3.0. For more information, see the [LICENSE](LICENSE) file.

