import cv2
import numpy as np
import heapq
import math

def process_image_to_grid(image_path, grid_size=25):
    """
    Converts the image into a binary obstacle grid and returns the original image for visualization.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found. Ensure '{image_path}' is in the directory.")
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (7, 7), 0)

    # Thresholding: Pixels > 190 (stones) become 0 (path). Pixels <= 190 (rocks/grass) become 255 (obstacles).
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    grass_mask = cv2.inRange(blurred, lower_green, upper_green)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    stone_mask = cv2.inRange(blurred, lower_white, upper_white)

    moveable_mask = cv2.bitwise_or(grass_mask, stone_mask)
    obstacle_mask = cv2.bitwise_not(moveable_mask)
    height, width = obstacle_mask.shape
    grid_rows = height // grid_size
    grid_cols = width // grid_size
    grid = np.zeros((grid_rows, grid_cols), dtype=int)

    for r in range(grid_rows):
        for c in range(grid_cols):
            cell = obstacle_mask[r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size]
            obstacle_ratio = np.sum(cell == 255) / (grid_size * grid_size)
            
            # If > 40% of the cell is rocks/shadows, mark as an obstacle
            if obstacle_ratio > 0.4:
                grid[r, c] = 1 
            else:
                grid[r, c] = 0 
                
    return img, grid

def heuristic(a, b):
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def a_star_search(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1])
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue 

                move_cost = math.sqrt(d[0]**2 + d[1]**2)
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None 

def visualize_grid_and_path(img, grid, grid_size, path=None):
    """
    Overlays a semi-transparent color-coded grid and the A* path onto the original image.
    """
    overlay = img.copy()
    rows, cols = grid.shape
    
    # 1. Color-code the grid cells
    for r in range(rows):
        for c in range(cols):
            top_left = (c * grid_size, r * grid_size)
            bottom_right = ((c + 1) * grid_size, (r + 1) * grid_size)
            
            if grid[r, c] == 1:
                # Obstacle = Red fill
                cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 255), -1) 
            else:
                # Moveable = Green fill
                cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), -1) 
                
    # Blend the color overlay with the original image (40% opacity for the colors)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 2. Draw white grid lines to separate the squares visually
    for r in range(rows + 1):
        cv2.line(img, (0, r * grid_size), (cols * grid_size, r * grid_size), (255, 255, 255), 1)
    for c in range(cols + 1):
        cv2.line(img, (c * grid_size, 0), (c * grid_size, rows * grid_size), (255, 255, 255), 1)
        
    # 3. Draw the A* path as a thick blue line connecting cell centers
    if path:
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            
            # Calculate the pixel center of each grid cell
            pt1 = (int((c1 + 0.5) * grid_size), int((r1 + 0.5) * grid_size))
            pt2 = (int((c2 + 0.5) * grid_size), int((r2 + 0.5) * grid_size))
            
            # Draw line segment and a circle at the node
            cv2.line(img, pt1, pt2, (255, 0, 0), 4) 
            cv2.circle(img, pt1, 5, (255, 0, 0), -1)
            
        # Draw the final node
        r_last, c_last = path[-1]
        pt_last = (int((c_last + 0.5) * grid_size), int((r_last + 0.5) * grid_size))
        cv2.circle(img, pt_last, 5, (0, 255, 255), -1) # Yellow circle for the goal
        
        # Mark the start node
        r_start, c_start = path[0]
        pt_start = (int((c_start + 0.5) * grid_size), int((r_start + 0.5) * grid_size))
        cv2.circle(img, pt_start, 5, (255, 0, 255), -1) # Purple circle for the start

    return img

if __name__ == "__main__":
    grid_size = 25
    img, grid = process_image_to_grid(r"C:\Users\awast\Downloads\terrain.webp", grid_size=grid_size)
    
    # Define start (bottom-left area) and goal (top-left area)
    start_node = (grid.shape[0] - 2, 1) 
    goal_node = (2, 1) 
    
    # Calculate the path
    path = None
    if grid[start_node[0], start_node[1]] == 1 or grid[goal_node[0], goal_node[1]] == 1:
        print("Warning: Start or Goal node is inside a red obstacle square. Cannot calculate path.")
    else:
        path = a_star_search(grid, start_node, goal_node)
    
    # Generate the visualization
    visualized_img = visualize_grid_and_path(img, grid, grid_size, path)
    
    # Save the output image so you can view it
    output_filename = r"C:\Users\awast\Downloads\terrain_path_analysis.jpg"
    cv2.imwrite(output_filename, visualized_img)
    print(f"Analysis complete. Saved visualization as '{output_filename}'.")