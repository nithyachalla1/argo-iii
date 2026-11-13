import cv2
import numpy as np
import heapq
import math

# Cost Function Definitions
COST_PAVEMENT = 1.0
COST_GRASS = 1.0       # Grass and flat stepping stones = 1.0 (Green)
COST_ROCKS = 30.0      # River pebbles = 30.0 (Red)
COST_WALLS = math.inf  # Background walls / patio structures = INF (Red)

def estimate_depth_factor(row, total_rows):
    """
    Simulates monocular camera depth perception using vertical image perspective.
    """
    normalized_dist = 1.0 - (row / float(total_rows))
    return 1.0 + (0.25 * normalized_dist)

def process_image_to_cost_grid(image_path, grid_size=35):
    """
    Separates terrain into:
    - Walls (INF): Vertical structures/patio above the ground horizon line
    - Grass (1): Green-dominant pixels (works even in dark pergola shadows)
    - Pavement/Stepping Stones (1): Smooth, bright, low-texture stone slabs
    - Rocks/Pebbles (30): High-texture river stones and gravel beds
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found. Ensure '{image_path}' is in the directory.")
        
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Split BGR channels as floats to detect green grass even in deep shadows
    b_ch, g_ch, r_ch = cv2.split(img.astype(np.float32))
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # 1. GRASS DETECTION (Handles both sunlit lawn and dark shaded lawn)
    # In RGB, grass has a stronger Green channel than Red or Blue, even when dark
    hsv_green = (h_ch >= 28) & (h_ch <= 90) & (s_ch >= 25) & (v_ch >= 15)
    rgb_green = (g_ch > r_ch * 1.05) & (g_ch > b_ch * 1.10) & (g_ch > 20)
    grass_mask = np.uint8((hsv_green | rgb_green) * 255)

    # 2. TEXTURE DETECTION (Separates smooth stepping stones from bumpy river pebbles)
    # River pebbles have dense, sharp edges and dark gaps between stones
    laplacian = np.abs(cv2.Laplacian(blurred_gray, cv2.CV_64F, ksize=3))
    # Smooth local texture energy over a 15x15 window so whole pebble beds light up
    texture_energy = cv2.blur(laplacian, (15, 15))

    # 3. PAVEMENT / STEPPING STONES (Bright, low saturation, low texture energy)
    is_bright_stone = (s_ch < 55) & (v_ch > 155)
    is_smooth = texture_energy < 18.0
    pavement_mask = np.uint8((is_bright_stone & is_smooth & ~ (grass_mask > 0)) * 255)

    grid_rows = height // grid_size
    grid_cols = width // grid_size
    
    cost_grid = np.zeros((grid_rows, grid_cols), dtype=float)
    terrain_labels = np.empty((grid_rows, grid_cols), dtype=object)

    cell_area = float(grid_size * grid_size)

    # In this camera perspective, the vertical patio wall/furniture sits in the top ~18% of the frame
    wall_horizon_row = max(3, int(grid_rows * 0.18))

    for r in range(grid_rows):
        for c in range(grid_cols):
            r_start, r_end = r * grid_size, (r + 1) * grid_size
            c_start, c_end = c * grid_size, (c + 1) * grid_size

            # Rule 1: Vertical Wall / Patio Structure at the top of the frame -> INF
            if r < wall_horizon_row:
                cost_grid[r, c] = COST_WALLS
                terrain_labels[r, c] = "Wall"
                continue

            grass_ratio = np.sum(grass_mask[r_start:r_end, c_start:c_end] == 255) / cell_area
            pave_ratio = np.sum(pavement_mask[r_start:r_end, c_start:c_end] == 255) / cell_area
            cell_texture = np.mean(texture_energy[r_start:r_end, c_start:c_end])

            # Rule 2: If cell is predominantly grass (even shaded grass) -> 1 (Green)
            if grass_ratio >= 0.40:
                cost_grid[r, c] = COST_GRASS
                terrain_labels[r, c] = "Grass"
            # Rule 3: If cell is predominantly flat stepping stone and low texture -> 1 (Green)
            elif pave_ratio >= 0.45 and cell_texture < 20.0:
                cost_grid[r, c] = COST_PAVEMENT
                terrain_labels[r, c] = "Pave"
            # Rule 4: Everything else on the ground plane is River Pebbles / Rocks -> 30 (Red)
            else:
                cost_grid[r, c] = COST_ROCKS
                terrain_labels[r, c] = "Rock"
                
    return img, cost_grid, terrain_labels

def heuristic(a, b):
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2) * COST_PAVEMENT

def a_star_search(cost_grid, start, goal):
    rows, cols = cost_grid.shape
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}
    
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
                
                if math.isinf(cell_cost):
                    continue 

                step_distance = math.sqrt(d[0]**2 + d[1]**2)
                depth_weight = estimate_depth_factor(nr, rows)
                move_cost = step_distance * cell_cost * depth_weight
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, math.inf

def visualize_cost_map_and_path(img, cost_grid, grid_size, path=None):
    overlay = img.copy()
    rows, cols = cost_grid.shape
    
    # 1. Color-code each cell: Both INF (Walls) and 30 (Rocks) are Red; 1 (Grass/Pave) is Green
    for r in range(rows):
        for c in range(cols):
            top_left = (c * grid_size, r * grid_size)
            bottom_right = ((c + 1) * grid_size, (r + 1) * grid_size)
            cost = cost_grid[r, c]
            
            if math.isinf(cost):
                color = (0, 0, 220)   # Red for Walls (INF)
            elif cost >= COST_ROCKS:
                color = (0, 0, 255)   # Red for Rocks (30)
            else:
                color = (0, 200, 0)   # Green for Grass & Stepping Stones (1)
                
            cv2.rectangle(overlay, top_left, bottom_right, color, -1)
                
    alpha = 0.42
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 2. Draw grid lines and cost numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.3, grid_size / 90.0)
    
    for r in range(rows):
        for c in range(cols):
            cost = cost_grid[r, c]
            cost_str = "INF" if math.isinf(cost) else f"{int(cost)}"
            
            text_x = int(c * grid_size + grid_size * 0.18)
            text_y = int(r * grid_size + grid_size * 0.62)
            
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
            
        r_last, c_last = path[-1]
        pt_last = (int((c_last + 0.5) * grid_size), int((r_last + 0.5) * grid_size))
        cv2.circle(img, pt_last, 7, (0, 255, 255), -1)
        
        r_start, c_start = path[0]
        pt_start = (int((c_start + 0.5) * grid_size), int((r_start + 0.5) * grid_size))
        cv2.circle(img, pt_start, 7, (255, 0, 255), -1)

    return img

if __name__ == "__main__":
    grid_size = 35
    image_path = r"C:\Users\awast\Downloads\terrain.webp"
    
    img, cost_grid, labels = process_image_to_cost_grid(image_path, grid_size=grid_size)
    
    # Start near bottom stepping stone, Goal just below the top wall horizon
    wall_horizon_row = max(3, int(cost_grid.shape[0] * 0.18))
    start_node = (cost_grid.shape[0] - 2, cost_grid.shape[1] // 3) 
    goal_node = (wall_horizon_row + 1, cost_grid.shape[1] // 2) 
    
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