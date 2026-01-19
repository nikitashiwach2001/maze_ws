#!/usr/bin/env python3
"""
Value Iteration for Maze Navigation
Computes optimal policy using Dynamic Programming
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime


class ValueIterationSolver:
    """
    Value Iteration for grid-based maze navigation
    """
    def __init__(self, maze_size=5.0, grid_resolution=0.5):
        """
        Args:
            maze_size: Size of square maze in meters (5x5)
            grid_resolution: Grid cell size in meters (0.5 = 10x10 grid)
        """
        self.maze_size = maze_size
        self.grid_resolution = grid_resolution
        self.grid_size = int(maze_size / grid_resolution)

        # Start and goal in grid coordinates
        self.start = (1, 1)  # (0.5, 0.5) in continuous space
        self.goal = (9, 9)   # (4.5, 4.5) in continuous space

        # Value function and policy
        self.V = np.zeros((self.grid_size, self.grid_size))
        self.policy = np.zeros((self.grid_size, self.grid_size, 2))

        # RL parameters
        self.gamma = 0.99  # Discount factor
        self.theta = 0.01  # Convergence threshold

        # Define maze walls (same as your SDF world)
        self.walls = self._create_maze_walls()

        print(f"Value Iteration Solver initialized")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Start: {self.start}, Goal: {self.goal}")
        print(f"Number of wall cells: {len(self.walls)}")

    def _create_maze_walls(self):
        """Define wall positions matching the maze world"""
        walls = set()

        # Outer boundary walls
        for i in range(self.grid_size):
            walls.add((i, 0))  # Bottom wall
            walls.add((i, self.grid_size - 1))  # Top wall
            walls.add((0, i))  # Left wall
            walls.add((self.grid_size - 1, i))  # Right wall

        # Internal maze walls - matching actual maze_world.sdf
        # Grid resolution is 0.5m, so each grid cell is 0.5m x 0.5m
        # Coordinate mapping: continuous position / 0.5 = grid index

        # wall_h1: at (1.5, 1), size 1m wide -> grid cells (3, 2)
        walls.add((3, 2))

        # wall_v1: at (2, 1.5), size 1m tall -> grid cells (4, 3)
        walls.add((4, 3))

        # wall_h2: at (3.5, 2), size 1m wide -> grid cells (7, 4)
        walls.add((7, 4))

        # wall_v2: at (3, 3.5), size 1m tall -> grid cells (6, 7)
        walls.add((6, 7))

        # wall_h3: at (1.5, 3), size 1m wide -> grid cells (3, 6)
        walls.add((3, 6))

        # wall_v3: at (4, 3.5), size 1m tall -> grid cells (8, 7)
        walls.add((8, 7))

        return walls

    def is_valid_state(self, x, y):
        """Check if state is valid (within bounds and not a wall)"""
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        return (x, y) not in self.walls

    def get_next_state(self, x, y, action):
        """
        Get next state given current state and action
        Actions: 0=right, 1=up, 2=left, 3=down
        """
        dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][action]
        nx, ny = x + dx, y + dy

        if self.is_valid_state(nx, ny):
            return nx, ny
        else:
            return x, y  # Stay in place if hit wall

    def get_reward(self, x, y, nx, ny):
        """Get reward for transition"""
        if (nx, ny) == self.goal:
            return 100.0  # Goal reward
        elif (nx, ny) in self.walls:
            return -50.0  # Wall penalty
        elif (nx, ny) == (x, y):
            return -10.0  # Penalty for staying in place
        else:
            return -1.0  # Step penalty

    def solve(self):
        """Run Value Iteration algorithm"""
        print("\nRunning Value Iteration...")

        iteration = 0
        while True:
            delta = 0
            iteration += 1

            # Iterate over all states
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    # Skip walls and goal
                    if (x, y) in self.walls or (x, y) == self.goal:
                        continue

                    v = self.V[x, y]

                    # Try all 4 actions and find max value
                    action_values = []
                    for action in range(4):
                        nx, ny = self.get_next_state(x, y, action)
                        reward = self.get_reward(x, y, nx, ny)
                        value = reward + self.gamma * self.V[nx, ny]
                        action_values.append(value)

                    # Update value function
                    self.V[x, y] = max(action_values)
                    delta = max(delta, abs(v - self.V[x, y]))

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: delta = {delta:.6f}")

            # Check convergence
            if delta < self.theta:
                break

        # Set goal value
        self.V[self.goal] = 100.0

        print(f"\nConverged in {iteration} iterations!")
        print(f"Optimal value at start: {self.V[self.start]:.2f}")
        print(f"Optimal value at goal: {self.V[self.goal]:.2f}")

        # Extract policy
        self._extract_policy()

        return self.V, self.policy

    def _extract_policy(self):
        """Extract optimal policy from value function"""
        action_dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) in self.walls or (x, y) == self.goal:
                    continue

                # Find best action
                best_action = 0
                best_value = -float('inf')

                for action in range(4):
                    nx, ny = self.get_next_state(x, y, action)
                    reward = self.get_reward(x, y, nx, ny)
                    value = reward + self.gamma * self.V[nx, ny]

                    if value > best_value:
                        best_value = value
                        best_action = action

                # Store policy as continuous action (for robot control)
                dx, dy = action_dirs[best_action]
                self.policy[x, y] = [dx * 0.5, dy * 0.5]  # Scale to robot velocity

    def get_action(self, x_continuous, y_continuous):
        """
        Get optimal action for continuous position
        Args:
            x_continuous, y_continuous: Position in meters
        Returns:
            action: [linear_velocity, angular_velocity]
        """
        # Convert to grid coordinates
        grid_x = int(x_continuous / self.grid_resolution)
        grid_y = int(y_continuous / self.grid_resolution)

        # Clip to valid range
        grid_x = np.clip(grid_x, 0, self.grid_size - 1)
        grid_y = np.clip(grid_y, 0, self.grid_size - 1)

        # Get policy action (velocity in x, y)
        vx, vy = self.policy[grid_x, grid_y]

        # Convert to robot action [linear, angular]
        # Simplified: just use forward velocity
        linear_vel = np.sqrt(vx**2 + vy**2)
        angular_vel = np.arctan2(vy, vx)  # Heading direction

        return np.array([linear_vel, angular_vel])

    def visualize(self, save_path='dp_solution.png'):
        """Visualize value function and policy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Plot value function
        V_display = self.V.copy()
        for wall in self.walls:
            V_display[wall] = np.nan

        im1 = ax1.imshow(V_display.T, origin='lower', cmap='viridis', interpolation='nearest')
        ax1.set_title('Value Function', fontsize=16)
        ax1.set_xlabel('Grid X')
        ax1.set_ylabel('Grid Y')
        ax1.plot(self.start[0], self.start[1], 'go', markersize=15, label='Start')
        ax1.plot(self.goal[0], self.goal[1], 'r*', markersize=20, label='Goal')
        ax1.legend()
        plt.colorbar(im1, ax=ax1, label='Value')

        # Plot policy arrows
        ax2.imshow(V_display.T, origin='lower', cmap='viridis', alpha=0.3, interpolation='nearest')
        ax2.set_title('Optimal Policy', fontsize=16)
        ax2.set_xlabel('Grid X')
        ax2.set_ylabel('Grid Y')

        # Draw policy arrows
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.walls and (x, y) != self.goal:
                    dx, dy = self.policy[x, y]
                    if dx != 0 or dy != 0:
                        ax2.arrow(x, y, dx*0.8, dy*0.8,
                                 head_width=0.2, head_length=0.1,
                                 fc='red', ec='red', alpha=0.7)

        ax2.plot(self.start[0], self.start[1], 'go', markersize=15, label='Start')
        ax2.plot(self.goal[0], self.goal[1], 'r*', markersize=20, label='Goal')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.close()

    def save_solution(self, filepath='dp_solution.json'):
        """Save value function and policy to file"""
        solution = {
            'value_function': self.V.tolist(),
            'policy': self.policy.tolist(),
            'grid_size': self.grid_size,
            'grid_resolution': self.grid_resolution,
            'start': self.start,
            'goal': self.goal,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(solution, f, indent=2)

        print(f"Solution saved to: {filepath}")


def main():
    """Main function to run Value Iteration"""
    print("="*70)
    print("DYNAMIC PROGRAMMING - VALUE ITERATION")
    print("="*70)

    # Create solver
    solver = ValueIterationSolver(maze_size=5.0, grid_resolution=0.5)

    # Solve
    V, policy = solver.solve()

    # Visualize
    solver.visualize(save_path='value_iteration_solution.png')

    # Save solution
    solver.save_solution(filepath='dp_solution.json')

    print("\n" + "="*70)
    print("DYNAMIC PROGRAMMING COMPLETE!")
    print("="*70)
    print("\nOptimal policy computed!")
    print("You can now use 'dp_navigator' to execute this policy in Gazebo")


if __name__ == '__main__':
    main()
