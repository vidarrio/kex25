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
├── data/                                 # Generated benchmark data
│   └── astar_vs_rl_throughput_*.json    
├── runs/                                 # Tensorboard logs
│   ├── phase_<phase>_dqn_YYYYMMDD-HHMM.pth
│   │   └── events.out.tfevents.*
│   └── ...
├── figures/                              # Generated figures
│   ├── rl_training_delivered.svg
│   ├── astar_vs_rl_throughput_*.svg
│   └── ...
└── src/
    ├── algorithms/
    │   ├── __init__.py
    │   ├── a_star.py
    │   ├── agent.py
    │   ├── benchmark.py
    │   ├── common.py
    │   ├── qnet.py
    │   ├── replay.py
    │   └── training.py
    ├── environment/
    │   ├── __init__.py
    │   ├── grid.py
    │   └── utils.py
    ├── tests/
    │   ├── __init__.py
    |   ├── test_env.py
    │   ├── test_a_star.py
    │   └── test_rl.py
    ├── models/
    │   ├── phase_<phase>_dqn_YYYYMMDD-HHMM.pth
    │   ├── phase_<phase>_dqn_YYYYMMDD-HHMM_best.pth
    │   └── ...
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
python src/main.py run_a_star
```

### Training the Reinforcement Learning Model
To train the reinforcement learning model, run the following command:
```bash
python src/main.py dqn_train
```

### Evaluating the Reinforcement Learning Model
To run the reinforcement learning model, run the following command:
```bash
python src/main.py dqn_run
```

### Display logs (tensorboard)
To display the logs of the training process, run the following command:
```bash
tensorboard --logdir=./runs/
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

## Dependencies
The project uses the following python dependencies:
- numpy==2.2.3
- scipy==1.15.2
- matplotlib==3.10.1
- torch==2.6.0
- pytest==8.3.5
- pettingzoo==1.24.3
- tensorboard==2.19.0
- seaborn==0.13.2

## Contact
For any questions or feedback, please contact us at
- Lucas Kristiansson: [luckri@hotmail.se](mailto:luckri@hotmail.se)
- Felix Winkelmann: [felix.winkelmann02@gmail.com](mailto:felix.winkelmann02@gmail.com)

## License
This project is licenced under GPL-3.0. For more information, see the [LICENSE](LICENSE) file.

