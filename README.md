# Maze Navigation with TD3 Reinforcement Learning

This package implements TD3 (twin D3) reinforcement learning for maze navigation with a differential drive robot.

## Robot Setup

Currently configured for **simple_robot**. For TurtleBot integration, see separate instructions below.

## Quick Start Guide

### Terminal 1: Build and Launch Gazebo

```bash
# Build the package
colcon build --packages-select maze_robot_qlearning
source install/setup.bash

# Launch Gazebo with maze world and simple_robot
ros2 launch maze_robot_qlearning maze_launch.py
```

The launch file spawns the simple_robot in the maze environment. If you want to use a different robot, edit the launch file before running.

### Terminal 2: Start TD7 Training

```bash
# Source the workspace
source install/setup.bash

# Run TD7 training
ros2 run maze_robot_qlearning train_td7
```

## Training Configuration

### Current Hyperparameters (Optimized for TD3)

- **Episodes**: 10,000
- **Actor Learning Rate**: 1e-4
- **Critic Learning Rate**: 3e-4
- **Encoder Learning Rate**: 3e-4
- **Network Dimensions**: hidden_dim=128, zs_dim=128, zsa_dim=128
- **Batch Size**: 256
- **Buffer Capacity**: 100,000 transitions
- **Policy Update Frequency**: Every 4 critic updates
- **Warmup Steps**: 3,000 steps (random actions)
- **Exploration Noise Decay**: 1.5

### Training Process

1. **Warmup Phase**: First 3,000 steps use random actions to collect diverse experiences
2. **Training Phase**: TD3 agent learns from replay buffer using LAP (Largest-Alpha-Priority) sampling
3. **Checkpoints**: Model saved every 50 episodes + best model + final model

### Model Checkpoints

Models are saved in:
- `td7_models/td7_best.pt` - Best performing model
- `td7_models/td7_episode_N.pt` - Checkpoint every 50 episodes
- `td7_models/td7_final.pt` - Final model after all episodes

Training logs saved in:
- `training_logs/td7_training_stats.json` - Episode rewards, losses, success rates

## Starting Fresh Training

**IMPORTANT**: If you want to start completely fresh training (delete previous learned policies):

```bash
# Delete model checkpoints and training logs
rm -rf td7_models/
rm -rf training_logs/

# Rebuild and restart training
colcon build --packages-select maze_robot_qlearning
source install/setup.bash
ros2 launch maze_robot_qlearning maze_launch.py  # Terminal 1
ros2 run maze_robot_qlearning train_td7          # Terminal 2
```

**Note**: Deleting only `training_logs/` is NOT enough - you must delete `td7_models/` to remove corrupted neural network weights.

## Environment Details

- **Observation Space**: 10 LiDAR readings (360° coverage, downsampled)
- **Action Space**: [linear_velocity, angular_velocity]
  - Linear: [0.0, 1.0] m/s
  - Angular: [-1.0, 1.0] rad/s
- **Collision Detection**: LiDAR-based with temporal median filtering (threshold: 0.22m)
- **Success Criteria**: Reach goal within 0.5m
- **Episode Timeout**: 300 steps
- **Stuck Detection**: Movement < 0.20m over 30 timesteps

## Monitoring Training

The training script logs:
- Every episode: Reward, steps, termination reason (success/collision/timeout/stuck)
- Every 10 episodes: Detailed statistics including:
  - Average reward (last 10 episodes)
  - Success rate (last 10 episodes)
  - Buffer size
  - Total steps

Example output:
```
✅ SUCCESS Episode 150: reward=87.32, steps=143, reason=goal_reached
❌ COLLISION Episode 151: reward=-8.45, steps=23, reason=collision
⏱️  TIMEOUT Episode 152: reward=12.67, steps=300, reason=timeout
```

## Testing Trained Models

Once you have trained a model, you can test it in Gazebo to visualize the learned policy.

### Terminal 1: Launch Gazebo

```bash
source install/setup.bash
ros2 launch maze_robot_qlearning maze_launch.py
```

### Terminal 2: Run Test Script

```bash
source install/setup.bash
ros2 run maze_robot_qlearning test_td
```

By default, the test script loads `td7_models/td7_best.pt` and runs 10 test episodes with the trained policy (no exploration noise).

### Testing Specific Checkpoints

To test a specific checkpoint, edit [test_td.py](maze_robot_qlearning/maze_robot_qlearning/test_td.py) line 93:

```python
# Test best model
MODEL_PATH = "td7_models/td7_best.pt"

# OR test specific checkpoint
MODEL_PATH = "td7_models/td7_episode_500.pt"

# OR test final model
MODEL_PATH = "td7_models/td7_final.pt"
```

### Test Parameters

You can modify the test behavior in [test_td.py](maze_robot_qlearning/maze_robot_qlearning/test_td.py) line 100:

```python
tester.run(num_episodes=10, max_steps=200)
```

- `num_episodes`: Number of test episodes to run
- `max_steps`: Maximum steps per episode (default: 200)

The test script runs with:
- **No exploration noise** (deterministic policy)
- **Visualization enabled** (Gazebo stays open)
- **Slower execution** (0.05s delay between steps for better visualization)

## TurtleBot Integration (Future)

To use TurtleBot instead of simple_robot:
1. Add LiDAR sensor to `urdf/turtlebot.urdf`
2. Replace deprecated Gazebo Classic controller with `libgazebo_ros_diff_drive.so`
3. Update collision threshold to 0.30m in `maze_environment.py`
4. Modify launch file to spawn TurtleBot
5. Adjust observation/action space if needed

## Troubleshooting

**Robot spinning at initial position**: Delete `td7_models/` directory to remove corrupted policy weights

**Collision on first step**: Check LiDAR data is being published on `/scan` topic

**Training not starting**: Verify Gazebo launched successfully and robot spawned

**Import errors**: Rebuild package with `colcon build --packages-select maze_robot_qlearning`
