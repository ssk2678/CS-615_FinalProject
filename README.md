# CS-615 Final Project

This project trains Deep Q Networks (DQNs) to control traffic signals in SUMO. It includes:

- PyTorch and handwritten NumPy (`NoTorch`) implementations.
- Single-intersection and multi-agent grid environments.
- `2x2` and `4x4` multi-agent scenarios.
- Standard SUMO-RL rewards and optional Zhang-style partial observations/rewards.

## Setup

SUMO, Python, PyTorch, NumPy, and SUMO-RL are required. The commands below use the existing Conda environment:

```bash
conda activate cs615-sumo
export SUMO_HOME=/usr/share/sumo
cd /home/tnmfdrt/Documents/Class/CS615/CS-615_FinalProject-main/Models
```

If the Python packages are missing, install them inside the active environment:

```bash
python -m pip install numpy torch sumo-rl
```

Training and testing currently use `use_gui=True`, so each command opens the SUMO GUI.

## Single-Agent Commands

### PyTorch

Train and test the standard model:

```bash
python tco_SingleAgent_TorchTrain.py
python tco_SingleAgent_TorchTest.py
```

Train and test with Zhang-style partial observations and rewards:

```bash
python tco_SingleAgent_TorchTrain.py --zhang
python tco_SingleAgent_TorchTest.py --zhang
```

### NoTorch (NumPy)

Train and test the standard model:

```bash
python tco_SingleAgent_NoTorchTrain.py
python tco_SingleAgent_NoTorchTest.py
```

Train and test with Zhang mode:

```bash
python tco_SingleAgent_NoTorchTrain.py --zhang
python tco_SingleAgent_NoTorchTest.py --zhang
```

## Multi-Agent Commands

Multi-agent commands require `--scenario 2x2` or `--scenario 4x4`. Add `--zhang` to both the training and corresponding test command to use Zhang mode.

### PyTorch

```bash
# Standard 2x2
python tco_MultiAgent_TorchTrain.py --scenario 2x2
python tco_MultiAgent_TorchTest.py --scenario 2x2

# Zhang 2x2
python tco_MultiAgent_TorchTrain.py --scenario 2x2 --zhang
python tco_MultiAgent_TorchTest.py --scenario 2x2 --zhang

# Standard 4x4
python tco_MultiAgent_TorchTrain.py --scenario 4x4
python tco_MultiAgent_TorchTest.py --scenario 4x4

# Zhang 4x4
python tco_MultiAgent_TorchTrain.py --scenario 4x4 --zhang
python tco_MultiAgent_TorchTest.py --scenario 4x4 --zhang
```

### NoTorch (NumPy)

```bash
# Standard 2x2
python tco_MultiAgent_NoTorchTrain.py --scenario 2x2
python tco_MultiAgent_NoTorchTest.py --scenario 2x2

# Zhang 2x2
python tco_MultiAgent_NoTorchTrain.py --scenario 2x2 --zhang
python tco_MultiAgent_NoTorchTest.py --scenario 2x2 --zhang

# Standard 4x4
python tco_MultiAgent_NoTorchTrain.py --scenario 4x4
python tco_MultiAgent_NoTorchTest.py --scenario 4x4

# Zhang 4x4
python tco_MultiAgent_NoTorchTrain.py --scenario 4x4 --zhang
python tco_MultiAgent_NoTorchTest.py --scenario 4x4 --zhang
```


## Generated Files

Training writes files into `Models/`:

- `.pt`: trained PyTorch weights.
- `.npz`: trained NoTorch/NumPy weights.
- `*_training_rewards.csv`: one summary row per episode.
- `*_training_step_metrics.csv`: detailed rows collected at every environment step and written at the end of training.

Single-agent step metrics include the action, reward, cumulative episode reward, epsilon, loss, and termination state. Multi-agent metrics contain one row per agent per step, including agent reward, total step reward, agent/mean epsilon, and loss.

Loss is blank during the first 31 steps because training begins after the replay buffer contains a full batch of 32 transitions. Re-running the same combination overwrites its existing weight and CSV files.
