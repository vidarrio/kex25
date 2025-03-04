# kex25

## Overview
This repository contains the code for our bachelor thesis at KTH, which focuses on robot planning in a warehouse environment. The project involves setting up a simulated grid environment, implementing an A* algorithm, and implementing a reinforcement learning algorithm. The goal is to compare the efficiency and performance of these two algorithms.

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
To run the tests, run the following command:
```bash

```

## Algorithms

### A* Algorithm
The A* algorithm is a pathfinding algorithm that is used to find the shortest path between two points. The algorithm uses a heuristic function to estimate the cost of reaching the goal from a given point. The algorithm is guaranteed to find the shortest path if the heuristic function is admissible, i.e., it never overestimates the cost of reaching the goal.

It is implemented in the [a_star.py](src/algorithms/a_star.py) file.

### Reinforcement Learning Algorithm
The reinforcement learning algorithm is a model-free machine learning algorithm that learns to make decisions by interacting with the environment. The algorithm learns a policy that maps states to actions by maximizing the expected cumulative reward. The algorithm uses a neural network to approximate the Q-function, which estimates the expected cumulative reward of taking an action in a given state.

It is implemented in the [rl.py](src/algorithms/rl.py) file.

## Contact
For any questions or feedback, please contact us at
- Lucas Kristiansson: [luckri@hotmail.se](mailto:luckri@hotmail.se)
- Felix Winkelmann: [felix.winkelmann02@gmail.com](mailto:felix.winkelmann02@gmail.com)

## License
This project is licenced under GPL-3.0. For more information, see the [LICENSE](LICENSE) file.

