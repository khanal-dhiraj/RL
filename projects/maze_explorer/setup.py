import random

## Module 0

# assuming it is a 2D grid world with 5x5 cells
maze_size = 5
states = [(x, y) for x in range(maze_size) for y in range(maze_size)]
# print(states)

actions = ["up", "down", "left", "right"]

# reward is 100 if the new state is the goal state, -1 for each step
def reward(state, goal_state):
    if state == goal_state:
        return 100
    else:
        return -1

def act(state, action, maze_size, walls):
    if action == "up":
        new_state = (state[0]-1, state[1])
    elif action == "down":
        new_state = (state[0]+1, state[1])
    elif action == "left":
        new_state = (state[0], state[1]-1)
    elif action == "right":
        new_state = (state[0], state[1]+1)

    # if the new state is out of bounds, return the current state
    if new_state[0] < 0 or new_state[0] >= maze_size or new_state[1] < 0 or new_state[1] >= maze_size:
        return state

    # if the new state is a wall, return the current state
    if walls[new_state[0]][new_state[1]]:
        return state

    return new_state

def generate_random_maze(size, wall_prob=0.3):
    walls = [[False for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i, j) != (0, 0) and (i, j) != (size-1, size-1):  # keep start/goal clear
                walls[i][j] = random.random() < wall_prob
    return walls


# Use it:

# walls are randomly placed in the maze
walls = generate_random_maze(maze_size)
# goal state is the bottom right corner
goal_state = (maze_size-1, maze_size-1)
# print the maze


