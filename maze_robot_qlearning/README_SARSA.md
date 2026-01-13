# SARSA On-Policy Reinforcement Learning for Maze Navigation

This package implements **SARSA** (State-Action-Reward-State-Action), an on-policy temporal difference learning algorithm, to train a robot to navigate through a maze and reach the goal.

## What is SARSA?

SARSA is an **on-policy** reinforcement learning algorithm that learns the Q-values for state-action pairs while following the same policy it's learning from. Unlike Q-Learning (off-policy), SARSA updates using the action actually taken by the epsilon-greedy policy, making it more conservative and safer for real-world applications.

### SARSA Update Rule
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```

Where:
- `s, a`: Current state and action
- `r`: Reward received
- `s', a'`: Next state and next action (actually taken by the policy)
- `α`: Learning rate (0.1)
- `γ`: Discount factor (0.99)

## Architecture Overview

### 1. Environment ([maze_environment.py](maze_robot_qlearning/maze_environment.py))
- **State Space**: 10×10 grid discretization of 5m×5m maze = 100 states
- **Action Space**: 4 discrete actions (Forward, Backward, Turn Left, Turn Right)
- **Rewards**:
  - +100: Reaching goal (distance < 0.3m to goal at 4.5, 4.5)
  - -50: Collision with walls (LIDAR distance < 0.25m)
  - -1: Each step (encourages efficiency)

### 2. SARSA Agent ([sarsa_agent.py](maze_robot_qlearning/sarsa_agent.py))
- **Q-Table**: 100 states × 4 actions
- **Policy**: Epsilon-greedy (on-policy exploration)
- **Hyperparameters**:
  - Learning rate (α): 0.1
  - Discount factor (γ): 0.99
  - Initial epsilon: 1.0 (100% exploration)
  - Epsilon decay: 0.995 per episode
  - Minimum epsilon: 0.01

### 3. Training Script ([train_sarsa.py](maze_robot_qlearning/train_sarsa.py))
- Manages training episodes
- Implements SARSA update loop
- Tracks statistics (rewards, success rate, collisions)
- Saves models periodically and best model

### 4. Evaluation Script ([evaluate_sarsa.py](maze_robot_qlearning/evaluate_sarsa.py))
- Tests trained policy (greedy, no exploration)
- Visualizes agent behavior in Gazebo
- Reports success rate and statistics

## Installation

### 1. Build the package
```bash
cd ~/robotics/maze_ws
colcon build --packages-select maze_robot_qlearning
source install/setup.bash
```

### 2. Verify installation
```bash
ros2 pkg list | grep maze_robot_qlearning
```

## Usage

### Training the SARSA Agent

Launch the training (starts Gazebo + training node):

```bash
ros2 launch maze_robot_qlearning train_launch.py
```

**What happens during training:**
1. Gazebo opens with the maze world and robot
2. Robot starts at position (0.5, 0.5)
3. Agent explores using epsilon-greedy policy
4. Q-table updates using SARSA rule after each action
5. Episodes run until goal reached, collision, or timeout (200 steps)
6. Epsilon decays over time (more exploitation, less exploration)
7. Models saved every 100 episodes in `sarsa_models/` directory

**Training output example:**
```
Episode  10/500 | Reward:  -25.00 | Steps:  42.5 | Success:  10.0% | Collision:  40.0% | Epsilon: 0.9512
Episode  20/500 | Reward:  -15.30 | Steps:  38.2 | Success:  20.0% | Collision:  30.0% | Epsilon: 0.9048
...
Episode 500/500 | Reward:   85.50 | Steps:  15.3 | Success:  90.0% | Collision:   5.0% | Epsilon: 0.0100
```

**Saved models:**
- `sarsa_models/sarsa_best.pkl`: Best performing model
- `sarsa_models/sarsa_final.pkl`: Final model after all episodes
- `sarsa_models/sarsa_checkpoint_epXXX.pkl`: Periodic checkpoints
- `sarsa_models/training_stats.npz`: Training statistics

### Evaluating the Trained Agent

Test the trained agent (uses greedy policy):

```bash
ros2 run maze_robot_qlearning evaluate_sarsa
```

**Optional arguments:**
```bash
# Specify model path
ros2 run maze_robot_qlearning evaluate_sarsa sarsa_models/sarsa_best.pkl

# Specify number of episodes
ros2 run maze_robot_qlearning evaluate_sarsa sarsa_models/sarsa_best.pkl 20

# Disable visualization delay
ros2 run maze_robot_qlearning evaluate_sarsa sarsa_models/sarsa_best.pkl 10 false 0.0
```

**Evaluation output:**
```
Episode 1 finished: SUCCESS! | Steps: 15 | Reward: 85.00 | Distance to goal: 0.12m
Episode 2 finished: SUCCESS! | Steps: 18 | Reward: 82.00 | Distance to goal: 0.08m
...
EVALUATION SUMMARY
==================
Episodes evaluated: 10
Successes: 9 (90.0%)
Collisions: 1 (10.0%)
Average reward: 84.50 ± 12.30
Average episode length: 16.2 ± 3.5 steps
```

### Manual Control (Testing Environment)

Test the environment manually with keyboard control:

```bash
# Terminal 1: Launch Gazebo
ros2 launch maze_robot_qlearning maze_launch.py

# Terminal 2: Run keyboard teleop
ros2 run maze_robot_qlearning keyboard_teleop
```

Controls: `W/A/S/D` for movement, `Ctrl+C` to quit

## Key Implementation Details

### Why SARSA is On-Policy

SARSA is **on-policy** because:
1. It uses the **same policy** for both action selection and learning
2. The next action `a'` in the update rule is selected by the epsilon-greedy policy
3. It learns the value of the policy it's actually following (including exploration)

**Contrast with Q-Learning (off-policy):**
- Q-Learning update: `Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]`
- Uses `max` over all actions (greedy) regardless of exploration policy
- Learns optimal policy while following exploratory policy

### State Representation

The 5×5m maze is discretized into a 10×10 grid:
- Each cell: 0.5m × 0.5m
- Robot position (x, y) → Grid state: `state = y_grid * 10 + x_grid`
- Example: Position (2.3, 1.7) → Grid (4, 3) → State 34

### Action Execution

Each action executes for 0.5 seconds:
- **Forward**: Linear velocity = 0.2 m/s
- **Backward**: Linear velocity = -0.2 m/s
- **Turn Left**: Angular velocity = 0.6 rad/s
- **Turn Right**: Angular velocity = -0.6 rad/s

### Episode Termination

Episodes end when:
1. **Success**: Distance to goal < 0.3m (+100 reward)
2. **Collision**: Min LIDAR distance < 0.25m (-50 reward)
3. **Out of Bounds**: Robot leaves maze boundaries (-50 reward)
4. **Timeout**: 200 steps exceeded (-1 per step)

## File Structure

```
maze_robot_qlearning/
├── maze_robot_qlearning/
│   ├── __init__.py
│   ├── maze_environment.py      # Environment wrapper (Gym-like interface)
│   ├── sarsa_agent.py            # SARSA agent with Q-table
│   ├── train_sarsa.py            # Training script
│   ├── evaluate_sarsa.py         # Evaluation script
│   └── keyboard_teleop.py        # Manual control
├── launch/
│   ├── maze_launch.py            # Launch Gazebo + robot
│   └── train_launch.py           # Launch training
├── worlds/
│   └── maze_world.sdf            # Gazebo maze world
├── urdf/
│   └── simple_robot.urdf         # Robot description
├── setup.py                      # Package setup
├── package.xml                   # ROS2 package manifest
└── README_SARSA.md               # This file
```

## Hyperparameter Tuning

Experiment with different hyperparameters in [train_sarsa.py](maze_robot_qlearning/train_sarsa.py:82-89):

```python
agent = SARSAAgent(
    num_states=env.get_num_states(),
    num_actions=env.get_num_actions(),
    learning_rate=0.1,          # α: Try 0.05, 0.2, 0.5
    discount_factor=0.99,       # γ: Try 0.9, 0.95, 0.99
    epsilon=1.0,                # Initial ε: Try 0.5, 0.8, 1.0
    epsilon_decay=0.995,        # Decay: Try 0.99, 0.995, 0.999
    epsilon_min=0.01            # Min ε: Try 0.0, 0.01, 0.05
)
```

## Troubleshooting

### Model not found during evaluation
```bash
# Make sure you've trained first
ros2 launch maze_robot_qlearning train_launch.py

# Check if model exists
ls sarsa_models/
```

### Robot not moving during training
- Check if Gazebo is running: `ps aux | grep gazebo`
- Verify topics: `ros2 topic list | grep cmd_vel`
- Check LIDAR: `ros2 topic echo /scan --once`

### Training too slow
- Reduce `num_episodes` in train_sarsa.py
- Increase `action_duration` in maze_environment.py (less accurate but faster)
- Run in headless mode (modify train_launch.py to set `gui:=false`)

### Poor success rate
- Train for more episodes (increase `num_episodes`)
- Tune hyperparameters (especially epsilon_decay)
- Check reward shaping in maze_environment.py:172

## Extending the Implementation

### Add More Actions
Edit [maze_environment.py](maze_robot_qlearning/maze_environment.py:14-18) Action enum:
```python
class Action(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    FORWARD_LEFT = 4   # New action
    FORWARD_RIGHT = 5  # New action
```

### Use LIDAR in State Representation
Modify `get_grid_state()` in [maze_environment.py](maze_robot_qlearning/maze_environment.py:93-98) to include obstacle information:
```python
def get_grid_state(self):
    # Grid position
    x_grid = int(np.clip(self.current_position[0] / self.cell_size, 0, self.grid_size - 1))
    y_grid = int(np.clip(self.current_position[1] / self.cell_size, 0, self.grid_size - 1))

    # Add obstacle detection (e.g., 2 bins: near/far)
    obstacle_near = 1 if self.min_lidar_distance < 0.5 else 0

    # State includes grid + obstacle info
    state = y_grid * self.grid_size * 2 + x_grid * 2 + obstacle_near
    return state
```

### Implement Q-Learning (Off-Policy)
Modify the update rule in train_sarsa.py to use max instead of next_action:
```python
# Q-Learning update (off-policy)
next_q = np.max(agent.q_table[next_state])  # Use max instead of next_action
agent.q_table[state, action] += agent.alpha * (reward + agent.gamma * next_q - agent.q_table[state, action])
```

## References

- **SARSA**: Rummery, G. A., & Niranjan, M. (1994). "On-line Q-learning using connectionist systems"
- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (2018), Chapter 6
- **ROS2**: https://docs.ros.org/en/humble/
- **Gazebo**: https://gazebosim.org/

## License

Apache License 2.0
