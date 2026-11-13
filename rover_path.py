import cv2
import numpy as np
import heapq
import math

# Cost Function Definitions
COST_PAVEMENT = 1.0
COST_GRASS = 5.0       # Optional intermediate cost for grass (set to 1.0 if identical to pavement)
COST_ROCKS = 30.0
COST_WALLS = math.inf  # Impassable obstacle

def estimate_depth_factor(row, total_rows):
    """
    Simulates monocular camera depth perception using vertical image perspective.
    In rover camera views, objects near the top (row -> 0) are farther away (near horizon),
    while objects near the bottom (row -> total_rows) are directly in front of the rover.
    Returns a confidence/scale weight (1.0 at bottom to 1.25 near horizon).
    """
    normalized_dist = 1.0 - (row / float(total_rows))
    return 1.0 + (0.25 * normalized_dist)

def process_image_to_cost_grid(image_path, grid_size=35):
    """
    Converts the camera image into a weighted cost grid based on HSV color/brightness 
    segmentation and perspective depth analysis.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found. Ensure '{image_path}' is in the directory.")
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (7, 7), 0)

    # 1. Grass detection (Green hues)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    grass_mask = cv2.inRange(blurred, lower_green, upper_green)

    # 2. Pavement / Stone path detection (High brightness, low saturation)
    lower_pavement = np.array([0, 0, 170])
    upper_pavement = np.array([180, 55, 255])
    pavement_mask = cv2.inRange(blurred, lower_pavement, upper_pavement)

    # 3. Walls / Deep Impassable Shadows (Very low brightness / dark structures)
    lower_wall = np.array([0, 0, 0])
    upper_wall = np.array([180, 255, 50])
    wall_mask = cv2.inRange(blurred, lower_wall, upper_wall)

    # 4. Rocks (Everything else: mid-tone grays, browns, textured obstacles)
    combined_known = cv2.bitwise_or(grass_mask, pavement_mask)
    combined_known = cv2.bitwise_or(combined_known, wall_mask)
    rock_mask = cv2.bitwise_not(combined_known)

    height, width, _ = img.shape
    grid_rows = height // grid_size
    grid_cols = width // grid_size
    
    # Use float grid to support math.inf for Walls
    cost_grid = np.zeros((grid_rows, grid_cols), dtype=float)
    terrain_labels = np.empty((grid_rows, grid_cols), dtype=object)

    cell_area = float(grid_size * grid_size)

    for r in range(grid_rows):
        for c in range(grid_cols):
            r_start, r_end = r * grid_size, (r + 1) * grid_size
            c_start, c_end = c * grid_size, (c + 1) * grid_size

            wall_ratio = np.sum(wall_mask[r_start:r_end, c_start:c_end] == 255) / cell_area
            rock_ratio = np.sum(rock_mask[r_start:r_end, c_start:c_end] == 255) / cell_area
            pave_ratio = np.sum(pavement_mask[r_start:r_end, c_start:c_end] == 255) / cell_area
            grass_ratio = np.sum(grass_mask[r_start:r_end, c_start:c_end] == 255) / cell_area

            # Classify cell based on dominant obstacle or surface hazard
            if wall_ratio > 0.35:
                cost_grid[r, c] = COST_WALLS
                terrain_labels[r, c] = "Wall"
            elif rock_ratio > 0.35:
                cost_grid[r, c] = COST_ROCKS
                terrain_labels[r, c] = "Rock"
            elif pave_ratio >= grass_ratio:
                cost_grid[r, c] = COST_PAVEMENT
                terrain_labels[r, c] = "Pave"
            else:
                # Grass defaults to pavement cost (1.0) to match strict 3-tier spec,
                # or change to COST_GRASS if you want the rover to prefer stone over grass.
                cost_grid[r, c] = COST_PAVEMENT
                terrain_labels[r, c] = "Pave"
                
    return img, cost_grid, terrain_labels

def heuristic(a, b):
    """
    Euclidean distance multiplied by the minimum possible traversal cost (COST_PAVEMENT = 1)
    so the heuristic remains admissible.
    """
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2) * COST_PAVEMENT

def a_star_search(cost_grid, start, goal):
    """
    Weighted A* Search that avoids infinite-cost walls and balances shorter paths 
    through rocks (cost 30) against longer detours on pavement (cost 1).
    """
    rows, cols = cost_grid.shape
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}
    
    # 8-way connectivity for smooth rover steering
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            total_cost = g_score[current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, total_cost

        for d in directions:
            nr, nc = current[0] + d[0], current[1] + d[1]
            neighbor = (nr, nc)
            
            if 0 <= nr < rows and 0 <= nc < cols:
                cell_cost = cost_grid[nr, nc]
                
                # Cannot traverse walls (cost == infinity)
                if math.isinf(cell_cost):
                    continue 

                # Geometric step distance (1.0 for cardinal, ~1.414 for diagonal)
                step_distance = math.sqrt(d[0]**2 + d[1]**2)
                
                # Incorporate monocular depth uncertainty factor
                depth_weight = estimate_depth_factor(nr, rows)
                
                # Total edge cost = distance * terrain cost * depth factor
                move_cost = step_distance * cell_cost * depth_weight
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, math.inf

def visualize_cost_map_and_path(img, cost_grid, grid_size, path=None):
    """
    Overlays the color-coded cost map, renders the cost value inside each grid cell,
    and plots the optimal rover trajectory.
    """
    overlay = img.copy()
    rows, cols = cost_grid.shape
    
    # 1. Color-code each cell by its traversal cost
    for r in range(rows):
        for c in range(cols):
            top_left = (c * grid_size, r * grid_size)
            bottom_right = ((c + 1) * grid_size, (r + 1) * grid_size)
            cost = cost_grid[r, c]
            
            if math.isinf(cost):
                # Walls (Infinity) = Dark Red / Crimson
                color = (0, 0, 180)
            elif cost >= COST_ROCKS:
                # Rocks (Cost 30) = Orange / Amber (high cost, passable in emergencies)
                color = (0, 140, 255)
            else:
                # Pavement (Cost 1) = Green (low cost, optimal)
                color = (0, 200, 0)
                
            cv2.rectangle(overlay, top_left, bottom_right, color, -1)
                
    # Blend color overlay with original image
    alpha = 0.42
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 2. Draw grid lines and print the exact cost inside each cell
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.3, grid_size / 90.0)
    
    for r in range(rows):
        for c in range(cols):
            cost = cost_grid[r, c]
            cost_str = "INF" if math.isinf(cost) else f"{int(cost)}"
            
            # Center text in cell
            text_x = int(c * grid_size + grid_size * 0.18)
            text_y = int(r * grid_size + grid_size * 0.62)
            
            # Draw black outline for readability, then white text
            cv2.putText(img, cost_str, (text_x, text_y), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, cost_str, (text_x, text_y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    for r in range(rows + 1):
        cv2.line(img, (0, r * grid_size), (cols * grid_size, r * grid_size), (220, 220, 220), 1)
    for c in range(cols + 1):
        cv2.line(img, (c * grid_size, 0), (c * grid_size, rows * grid_size), (220, 220, 220), 1)
        
    # 3. Draw the A* path
    if path:
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            
            pt1 = (int((c1 + 0.5) * grid_size), int((r1 + 0.5) * grid_size))
            pt2 = (int((c2 + 0.5) * grid_size), int((r2 + 0.5) * grid_size))
            
            cv2.line(img, pt1, pt2, (255, 50, 0), 4) 
            cv2.circle(img, pt1, 4, (255, 200, 0), -1)
            
        # Goal Node (Yellow) & Start Node (Purple)
        r_last, c_last = path[-1]
        pt_last = (int((c_last + 0.5) * grid_size), int((r_last + 0.5) * grid_size))
        cv2.circle(img, pt_last, 7, (0, 255, 255), -1)
        
        r_start, c_start = path[0]
        pt_start = (int((c_start + 0.5) * grid_size), int((r_start + 0.5) * grid_size))
        cv2.circle(img, pt_start, 7, (255, 0, 255), -1)

    return img

if __name__ == "__main__":
    # Slightly larger default grid_size (35px) makes the printed cost numbers easy to read
    grid_size = 35
    image_path = r"C:\Users\awast\Downloads\terrain.webp"
    
    img, cost_grid, labels = process_image_to_cost_grid(image_path, grid_size=grid_size)
    
    # Define Rover Start (bottom center-left) and Goal (top horizon area)
    start_node = (cost_grid.shape[0] - 2, cost_grid.shape[1] // 2) 
    goal_node = (2, cost_grid.shape[1] // 2) 
    
    path = None
    if math.isinf(cost_grid[start_node[0], start_node[1]]) or math.isinf(cost_grid[goal_node[0], goal_node[1]]):
        print("Warning: Start or Goal node is inside an impassable Wall (INF). Cannot calculate path.")
    else:
        path, total_cost = a_star_search(cost_grid, start_node, goal_node)
        if path:
            print(f"Path found! Steps: {len(path)} | Total Cumulative Cost: {total_cost:.2f}")
        else:
            print("No valid path found—target is completely walled off.")
    
    visualized_img = visualize_cost_map_and_path(img, cost_grid, grid_size, path)
    
    output_filename = r"C:\Users\awast\Downloads\terrain_cost_map_analysis.jpg"
    cv2.imwrite(output_filename, visualized_img)
    print(f"Saved cost map visualization as '{output_filename}'.")