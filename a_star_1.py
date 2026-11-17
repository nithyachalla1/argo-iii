import matplotlib.pyplot as plt
import numpy as np

# Helper function to estimate the remaining distance to the goal (using grid steps)
def get_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# The A* Algorithm
def a_star_search(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0])

    # List of nodes we still need to explore.
    # Format: [total_estimated_cost, position]
    open_list = [[0, start]]

    # Dictionaries to keep track of distances and paths
    parent_map = {} # Saves where we came from for each cell
    g_score = {start: 0} # Saves the number of steps taken from start to each cell

    while len(open_list) > 0:
        # 1. Sort the list so that the node with the lowest estimated cost is at the beginning
        open_list.sort()
        current_node = open_list.pop(0) # Get the best node
        current_pos = current_node[1]

        # If we reached the goal, build the path list backwards
        if current_pos == goal:
            path = []
            while current_pos in parent_map:
                path.append(current_pos)
                current_pos = parent_map[current_pos]
            path.append(start)
            return path[::-1] # Reverse the path so it goes from start to goal

        # 2. Check all 4 neighbor cells (Up, Down, Left, Right)
        r, c = current_pos
        neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

        for nr, nc in neighbors:
            # Make sure the neighbor cell is inside the grid boundaries
            if 0 <= nr < rows and 0 <= nc < cols:
                # Skip if it is an obstacle
                if grid[nr][nc] == 1:
                    continue

                # Calculate the cost to reach this neighbor
                new_g = g_score[current_pos] + 1

                # If this is a better (shorter) path to the neighbor, save it
                if new_g < g_score.get((nr, nc), float('inf')):
                    parent_map[(nr, nc)] = current_pos
                    g_score[(nr, nc)] = new_g
                    f_score = new_g + get_distance((nr, nc), goal)
                    open_list.append([f_score, (nr, nc)])

    return None # Return None if no path exists

# Helper function to plot our grid maps
def visualize_grid(grid, start, goal, path=None, title="Grid"):
    rows = len(grid)
    cols = len(grid[0])

    # Create an RGB image grid filled with white pixels by default
    # Shape is (rows, cols, 3) because each pixel has Red, Green, Blue channels
    img = np.ones((rows, cols, 3))

    # Fill color for obstacles (Black: R=0, G=0, B=0)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                img[r, c] = [0, 0, 0]

    # Fill color for the path (Blue: R=0.2, G=0.4, B=1.0)
    if path:
        for r, c in path:
            img[r, c] = [0.2, 0.4, 1.0]

    # Fill color for start (Green: R=0, G=0.8, B=0) and goal (Red: R=1, G=0, B=0)
    img[start[0], start[1]] = [0, 0.8, 0]
    img[goal[0], goal[1]] = [1, 0, 0]

    # Plot using Matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)

    # Make nice grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    plt.title(title, fontsize=12)
    plt.show()

"""### Test Scenarios"""

# Define coordinates
start = (0, 0)
goal = (4, 4)

# Scenario 1: Grid with obstacles (Provided Map)
grid_with_obstacles = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

path_1 = a_star_search(grid_with_obstacles, start, goal)
print(f"Scenario 1 (Grid with Obstacles):")
if path_1:
    print(f"Path found! Total path length (steps): {len(path_1) - 1}")
    visualize_grid(grid_with_obstacles, start, goal, path_1, "Scenario 1: Grid with Obstacles")
else:
    print("No path found.")
    visualize_grid(grid_with_obstacles, start, goal, None, "Scenario 1: Grid with Obstacles (Failed)")

# Scenario 2: Empty Grid
empty_grid = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

path_2 = a_star_search(empty_grid, start, goal)
print(f"Scenario 2 (Empty Grid):")
if path_2:
    print(f"Path found! Total path length (steps): {len(path_2) - 1}")
    visualize_grid(empty_grid, start, goal, path_2, "Scenario 2: Empty Grid")
else:
    print("No path found.")
    visualize_grid(empty_grid, start, goal, None, "Scenario 2: Empty Grid (Failed)")

# Scenario 3: Unreachable Goal (Goal is fully enclosed by obstacles)
unreachable_grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 1, 0]
]

path_3 = a_star_search(unreachable_grid, start, goal)
print(f"Scenario 3 (Unreachable Goal):")
if path_3:
    print(f"Path found! Total path length (steps): {len(path_3) - 1}")
    visualize_grid(unreachable_grid, start, goal, path_3, "Scenario 3: Unreachable Goal")
else:
    print("Path not found! Failure handled gracefully.")
    visualize_grid(unreachable_grid, start, goal, None, "Scenario 3: Unreachable Goal (Failed)")