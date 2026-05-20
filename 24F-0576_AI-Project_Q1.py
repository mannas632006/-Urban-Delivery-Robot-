"""
MODULE 1 - Intelligent Urban Delivery Robot

Search algorithms implemented:
  - Breadth First Search (BFS)
  - Depth First Search (DFS)
  - Uniform Cost Search (UCS)
  - Greedy Best First Search
  - A* Search

Author: Muhammad Anas
"""

import random
import time
import heapq
import copy
import os

# Check Check kar ke pygame install he ya nai for visualization
PYGAME_AVAILABLE = False
try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    pass

if not PYGAME_AVAILABLE:
    print("pygame not installed. Will use text-based visualization.")
    print("Install with: pip install pygame")


# Grid aur Envirnment

# Cell types
ROAD = 0
BUILDING = 1
DELIVERY = 2
TRAFFIC = 3
BASE_STATION = 4

CELL_NAMES = {
    ROAD: "Road",
    BUILDING: "Building",
    DELIVERY: "Delivery",
    TRAFFIC: "Traffic Zone",
    BASE_STATION: "Base Station",
}


def create_environment(rows=15, cols=15, num_deliveries=5, num_buildings=35,
                       num_traffic=15, seed=None):
    """
    Build the 15x15 grid environment.

    Returns:
        grid        – 2D list of cell types
        cost_grid   – 2D list of traversal costs
        base        – (row, col) of the base station
        deliveries  – list of (row, col) for delivery destinations
    """
    if seed is not None:
        random.seed(seed)

    grid = [[ROAD for _ in range(cols)] for _ in range(rows)]
    cost_grid = [[0 for _ in range(cols)] for _ in range(rows)]

    # Assign road costs first (1 se 5 for every cell)
    for r in range(rows):
        for c in range(cols):
            cost_grid[r][c] = random.randint(1, 5)

    # Place base station somewhere near the centerish area
    base_r = random.randint(5, 9)
    base_c = random.randint(5, 9)
    grid[base_r][base_c] = BASE_STATION
    cost_grid[base_r][base_c] = 0  # no cost to stand at base

    occupied = {(base_r, base_c)}

    # Place buildings (obstacles) - Ye sai nai chala tha kal dekhen ge ise
    placed = 0
    attempts = 0
    while placed < num_buildings and attempts < 500:
        r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
        if (r, c) not in occupied:
            grid[r][c] = BUILDING
            cost_grid[r][c] = -1  # impassable
            occupied.add((r, c))
            placed += 1
        attempts += 1

    # Place traffic zones on road cells
    placed = 0
    attempts = 0
    while placed < num_traffic and attempts < 500:
        r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
        if (r, c) not in occupied and grid[r][c] == ROAD:
            grid[r][c] = TRAFFIC
            cost_grid[r][c] = random.randint(10, 20)
            occupied.add((r, c))
            placed += 1
        attempts += 1

    # Place delivery locations on road cells
    deliveries = []
    placed = 0
    attempts = 0
    while placed < num_deliveries and attempts < 500:
        r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
        if (r, c) not in occupied and grid[r][c] == ROAD:
            grid[r][c] = DELIVERY
            # delivery cells keep their normal road cost
            occupied.add((r, c))
            deliveries.append((r, c))
            placed += 1
        attempts += 1

    base = (base_r, base_c)
    return grid, cost_grid, base, deliveries


def get_neighbors(grid, cost_grid, pos):
    """Return walkable neighbors of pos with their step cost."""
    rows, cols = len(grid), len(grid[0])
    r, c = pos
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != BUILDING:
            neighbors.append(((nr, nc), cost_grid[nr][nc]))
    return neighbors


def manhattan_distance(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a, b):
    """Euclidean distance heuristic."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# SEARCH ALGOS :- 

def bfs(grid, cost_grid, start, goal):
    """
    Breadth First Search - explores level by level.
    Does NOT consider edge costs, so the path it finds is shortest
    in number of steps, not necessarily cheapest.
    """
    from collections import deque

    visited = set()
    queue = deque()
    queue.append((start, [start]))
    visited.add(start)
    nodes_explored = 0

    while queue:
        current, path = queue.popleft()
        nodes_explored += 1

        if current == goal:
            # compute actual cost of the path
            total_cost = sum(cost_grid[p[0]][p[1]] for p in path[1:])
            return path, total_cost, nodes_explored

        for neighbor, step_cost in get_neighbors(grid, cost_grid, current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, float('inf'), nodes_explored


def dfs(grid, cost_grid, start, goal):
    """
    Depth First Search - goes deep before backtracking.
    Uses a stack. Not optimal for cost but explores differently. Acha Acha Bap no na sekha
    """
    visited = set()
    stack = [(start, [start])]
    nodes_explored = 0

    while stack:
        current, path = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1

        if current == goal:
            total_cost = sum(cost_grid[p[0]][p[1]] for p in path[1:])
            return path, total_cost, nodes_explored

        for neighbor, step_cost in get_neighbors(grid, cost_grid, current):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return None, float('inf'), nodes_explored


def ucs(grid, cost_grid, start, goal):
    """
    Uniform Cost Search - always expands the cheapest accumulated
    cost node first. Guarantees optimal cost path.
    """
    visited = set()
    # priority queue: (accumulated_cost, counter, position, path) Shukar he 
    counter = 0
    pq = [(0, counter, start, [start])]
    nodes_explored = 0

    while pq:
        acc_cost, _, current, path = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1

        if current == goal:
            return path, acc_cost, nodes_explored

        for neighbor, step_cost in get_neighbors(grid, cost_grid, current):
            if neighbor not in visited:
                counter += 1
                heapq.heappush(pq, (acc_cost + step_cost, counter, neighbor,
                                    path + [neighbor]))

    return None, float('inf'), nodes_explored


def greedy_best_first(grid, cost_grid, start, goal):
    """
    Greedy Best First Search - picks the node that looks closest
    to the goal (Manhattan distance). Fast but not always optimal.
    """
    visited = set()
    counter = 0
    h = manhattan_distance(start, goal)
    pq = [(h, counter, start, [start])]
    nodes_explored = 0

    while pq:
        _, _, current, path = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1

        if current == goal:
            total_cost = sum(cost_grid[p[0]][p[1]] for p in path[1:])
            return path, total_cost, nodes_explored

        for neighbor, step_cost in get_neighbors(grid, cost_grid, current):
            if neighbor not in visited:
                counter += 1
                h = manhattan_distance(neighbor, goal)
                heapq.heappush(pq, (h, counter, neighbor,
                                    path + [neighbor]))

    return None, float('inf'), nodes_explored


def a_star(grid, cost_grid, start, goal):
    """
    A* Search - combines actual cost so far (g) with heuristic
    estimate to goal (h). Optimal and usually efficient.
    Uses Manhattan distance as the heuristic.
    """
    visited = set()
    counter = 0
    g_score = {start: 0}
    h = manhattan_distance(start, goal)
    pq = [(h, counter, start, [start])]
    nodes_explored = 0

    while pq:
        f, _, current, path = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1

        if current == goal:
            return path, g_score[current], nodes_explored

        for neighbor, step_cost in get_neighbors(grid, cost_grid, current):
            if neighbor not in visited:
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    counter += 1
                    h = manhattan_distance(neighbor, goal)
                    heapq.heappush(pq, (tentative_g + h, counter, neighbor,
                                        path + [neighbor]))

    return None, float('inf'), nodes_explored


# Map algorithm names to functions
ALGORITHMS = {
    "BFS": bfs,
    "DFS": dfs,
    "UCS": ucs,
    "Greedy Best First": greedy_best_first,
    "A*": a_star,
}


# TEXT-BASED VISUALIZATION

CELL_CHARS = {
    ROAD: '.',
    BUILDING: '#',
    DELIVERY: 'D',
    TRAFFIC: '~',
    BASE_STATION: 'B',
}


def print_grid(grid, cost_grid, path=None, robot_pos=None):
    """Print the grid to the console with the path marked."""
    path_set = set(path) if path else set()
    rows, cols = len(grid), len(grid[0])

    # column numbers
    header = "    " + " ".join(f"{c:2d}" for c in range(cols))
    print(header)
    print("   " + "-" * (cols * 3 + 1))

    for r in range(rows):
        row_str = f"{r:2d} |"
        for c in range(cols):
            if robot_pos and (r, c) == robot_pos:
                row_str += " R "
            elif (r, c) in path_set and grid[r][c] not in (BASE_STATION, DELIVERY):
                row_str += " * "
            else:
                ch = CELL_CHARS.get(grid[r][c], '?')
                row_str += f" {ch} "
        row_str += "|"
        print(row_str)

    print("   " + "-" * (cols * 3 + 1))
    print()
    print("Legend:  . = Road   # = Building   D = Delivery   ~ = Traffic   B = Base   * = Path   R = Robot")


# PYGAME GRAPHICAL INTERFACE

# Colors for the GUI
COLORS = {
    ROAD:         (200, 200, 200),  # light gray
    BUILDING:     (60, 60, 60),     # dark gray
    DELIVERY:     (50, 180, 80),    # green
    TRAFFIC:      (230, 180, 50),   # amber/yellow
    BASE_STATION: (50, 120, 220),   # blue
    'path':       (180, 100, 220),  # purple trail
    'robot':      (220, 50, 50),    # red robot
    'grid_line':  (170, 170, 170),  # grid lines
    'text':       (30, 30, 30),     # dark text
    'bg':         (245, 245, 240),  # off-white background
    'panel':      (255, 255, 255),  # white panel
    'highlight':  (255, 230, 100),  # highlight
}


def run_pygame_visualization(grid, cost_grid, base, deliveries):
    """
    Launch a pygame window that lets you watch the robot
    deliver packages using different algorithms.
    """
    pygame.init()

    rows, cols = len(grid), len(grid[0])
    cell_size = 42
    grid_width = cols * cell_size
    grid_height = rows * cell_size
    panel_width = 380
    window_width = grid_width + panel_width + 30
    window_height = max(grid_height + 80, 700)

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Intelligent Urban Delivery Robot")

    # Fonts
    try:
        font_small = pygame.font.SysFont("consolas", 13)
        font_med = pygame.font.SysFont("consolas", 15)
        font_large = pygame.font.SysFont("consolas", 18, bold=True)
        font_title = pygame.font.SysFont("consolas", 20, bold=True)
    except:
        font_small = pygame.font.Font(None, 16)
        font_med = pygame.font.Font(None, 18)
        font_large = pygame.font.Font(None, 22)
        font_title = pygame.font.Font(None, 24)

    clock = pygame.time.Clock()

    # State variables
    grid_offset_x = 15
    grid_offset_y = 50

    algo_names = list(ALGORITHMS.keys())
    selected_algo = 0  # index into algo_names
    animation_speed = 10  # frames between steps
    running = True

    # Simulation state
    sim_running = False
    sim_paused = False
    current_delivery_idx = 0
    robot_pos = base
    current_path = []
    path_index = 0
    visited_trail = []  # cells the robot has passed through
    frame_counter = 0

    # Completed deliveries tracking
    completed_deliveries = []
    remaining_deliveries = list(deliveries)

    # Performance metrics for display
    results_log = []  # list of dicts with per-delivery stats
    total_cost_so_far = 0
    total_nodes_explored = 0
    total_time_elapsed = 0.0

    # Buttons (simple rectangles)
    panel_x = grid_width + 25
    btn_start = pygame.Rect(panel_x, 500, 160, 35)
    btn_reset = pygame.Rect(panel_x + 180, 500, 160, 35)
    btn_speed_up = pygame.Rect(panel_x + 270, 555, 70, 28)
    btn_speed_down = pygame.Rect(panel_x, 555, 70, 28)

    algo_buttons = []
    for i, name in enumerate(algo_names):
        rect = pygame.Rect(panel_x, 130 + i * 38, 340, 30)
        algo_buttons.append((rect, i))

    def start_next_delivery():
        nonlocal sim_running, current_path, path_index, frame_counter
        nonlocal current_delivery_idx, total_cost_so_far
        nonlocal total_nodes_explored, total_time_elapsed

        if not remaining_deliveries:
            sim_running = False
            return

        goal = remaining_deliveries[0]
        algo_func = ALGORITHMS[algo_names[selected_algo]]

        t_start = time.perf_counter()
        path, cost, nodes = algo_func(grid, cost_grid, robot_pos, goal)
        t_end = time.perf_counter()
        elapsed = t_end - t_start

        if path is None:
            # no path found – skip this delivery
            results_log.append({
                'delivery': current_delivery_idx + 1,
                'target': goal,
                'status': 'NO PATH',
                'cost': 0,
                'nodes': nodes,
                'time': elapsed,
            })
            remaining_deliveries.pop(0)
            current_delivery_idx += 1
            start_next_delivery()
            return

        current_path = path
        path_index = 0
        frame_counter = 0
        sim_running = True

        total_cost_so_far += cost
        total_nodes_explored += nodes
        total_time_elapsed += elapsed

        results_log.append({
            'delivery': current_delivery_idx + 1,
            'target': goal,
            'status': 'OK',
            'cost': cost,
            'nodes': nodes,
            'time': elapsed,
        })

    def reset_simulation():
        nonlocal sim_running, sim_paused, robot_pos, current_path, path_index
        nonlocal visited_trail, completed_deliveries, remaining_deliveries
        nonlocal current_delivery_idx, results_log
        nonlocal total_cost_so_far, total_nodes_explored, total_time_elapsed

        sim_running = False
        sim_paused = False
        robot_pos = base
        current_path = []
        path_index = 0
        visited_trail = []
        completed_deliveries = []
        remaining_deliveries = list(deliveries)
        current_delivery_idx = 0
        results_log = []
        total_cost_so_far = 0
        total_nodes_explored = 0
        total_time_elapsed = 0.0

    def draw_grid_cells():
        for r in range(rows):
            for c in range(cols):
                x = grid_offset_x + c * cell_size
                y = grid_offset_y + r * cell_size
                cell_type = grid[r][c]

                # Check if this is a completed delivery
                if (r, c) in completed_deliveries:
                    color = (140, 200, 140)  # faded green
                elif (r, c) in visited_trail and cell_type == ROAD:
                    color = COLORS['path']
                else:
                    color = COLORS.get(cell_type, COLORS[ROAD])

                pygame.draw.rect(screen, color,
                                 (x, y, cell_size, cell_size))
                pygame.draw.rect(screen, COLORS['grid_line'],
                                 (x, y, cell_size, cell_size), 1)

                # Draw cost number on road/traffic cells
                if cell_type in (ROAD, TRAFFIC, DELIVERY) and cost_grid[r][c] > 0:
                    cost_text = font_small.render(str(cost_grid[r][c]), True,
                                                  (100, 100, 100))
                    screen.blit(cost_text, (x + 3, y + 3))

                # Labels on special cells
                if cell_type == BASE_STATION:
                    label = font_med.render("B", True, (255, 255, 255))
                    screen.blit(label, (x + cell_size // 2 - 5,
                                        y + cell_size // 2 - 7))
                elif cell_type == DELIVERY:
                    label = font_med.render("D", True, (255, 255, 255))
                    screen.blit(label, (x + cell_size // 2 - 5,
                                        y + cell_size // 2 - 7))
                elif cell_type == BUILDING:
                    label = font_small.render("#", True, (120, 120, 120))
                    screen.blit(label, (x + cell_size // 2 - 4,
                                        y + cell_size // 2 - 6))

    def draw_path_preview():
        """Draw the planned path as dots."""
        if current_path and sim_running:
            for i, (pr, pc) in enumerate(current_path):
                if i > path_index:
                    x = grid_offset_x + pc * cell_size + cell_size // 2
                    y = grid_offset_y + pr * cell_size + cell_size // 2
                    pygame.draw.circle(screen, (200, 140, 255), (x, y), 4)

    def draw_robot():
        """Draw the robot as a red circle with a small outline."""
        rr, rc = robot_pos
        cx = grid_offset_x + rc * cell_size + cell_size // 2
        cy = grid_offset_y + rr * cell_size + cell_size // 2
        pygame.draw.circle(screen, (180, 30, 30), (cx, cy), cell_size // 3 + 2)
        pygame.draw.circle(screen, COLORS['robot'], (cx, cy), cell_size // 3)
        # small "eye" to show direction
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy - 3), 4)
        pygame.draw.circle(screen, (30, 30, 30), (cx, cy - 3), 2)

    def draw_panel():
        """Draw the right-side information panel."""
        # Panel background
        panel_rect = pygame.Rect(panel_x - 10, 40, panel_width + 10,
                                 window_height - 50)
        pygame.draw.rect(screen, COLORS['panel'], panel_rect, border_radius=8)
        pygame.draw.rect(screen, (200, 200, 200), panel_rect, 1, border_radius=8)

        # Title
        title = font_title.render("Delivery Robot Control", True, COLORS['text'])
        screen.blit(title, (panel_x, 55))

        # Algorithm selection
        algo_label = font_large.render("Select Algorithm:", True, COLORS['text'])
        screen.blit(algo_label, (panel_x, 100))

        for rect, idx in algo_buttons:
            if idx == selected_algo:
                pygame.draw.rect(screen, (50, 120, 220), rect, border_radius=5)
                text_color = (255, 255, 255)
            else:
                pygame.draw.rect(screen, (230, 230, 230), rect, border_radius=5)
                pygame.draw.rect(screen, (180, 180, 180), rect, 1, border_radius=5)
                text_color = COLORS['text']
            label = font_med.render(algo_names[idx], True, text_color)
            screen.blit(label, (rect.x + 12, rect.y + 7))

        # Status info
        y_offset = 340
        status_label = font_large.render("Status:", True, COLORS['text'])
        screen.blit(status_label, (panel_x, y_offset))
        y_offset += 25

        if sim_running:
            state_txt = "DELIVERING..." if not sim_paused else "PAUSED"
        elif remaining_deliveries:
            state_txt = "READY"
        else:
            state_txt = "ALL DONE"

        state = font_med.render(f"  State: {state_txt}", True, COLORS['text'])
        screen.blit(state, (panel_x, y_offset))
        y_offset += 22

        delivered_count = len(completed_deliveries)
        info_lines = [
            f"  Deliveries: {delivered_count} / {len(deliveries)}",
            f"  Total Cost: {total_cost_so_far}",
            f"  Nodes Explored: {total_nodes_explored}",
            f"  Time: {total_time_elapsed:.4f}s",
            f"  Speed: {animation_speed} (frames/step)",
        ]
        for line in info_lines:
            txt = font_med.render(line, True, COLORS['text'])
            screen.blit(txt, (panel_x, y_offset))
            y_offset += 20

        # Delivery log
        y_offset += 10
        log_label = font_large.render("Delivery Log:", True, COLORS['text'])
        screen.blit(log_label, (panel_x, y_offset))
        y_offset += 22

        for entry in results_log[-5:]:  # show last 5
            status_str = entry['status']
            d_num = entry['delivery']
            c = entry['cost']
            n = entry['nodes']
            log_line = f"  #{d_num}: cost={c}, nodes={n}, {status_str}"
            txt = font_small.render(log_line, True, (80, 80, 80))
            screen.blit(txt, (panel_x, y_offset))
            y_offset += 17

        # Buttons
        # Start button
        btn_color = (50, 160, 80) if not sim_running else (160, 160, 160)
        pygame.draw.rect(screen, btn_color, btn_start, border_radius=6)
        start_txt = font_med.render("Start Delivery", True, (255, 255, 255))
        screen.blit(start_txt, (btn_start.x + 20, btn_start.y + 8))

        # Reset button
        pygame.draw.rect(screen, (200, 60, 60), btn_reset, border_radius=6)
        reset_txt = font_med.render("Reset", True, (255, 255, 255))
        screen.blit(reset_txt, (btn_reset.x + 50, btn_reset.y + 8))

        # Speed controls
        speed_label = font_med.render(f"Animation Speed:", True, COLORS['text'])
        screen.blit(speed_label, (panel_x + 80, 558))

        pygame.draw.rect(screen, (100, 100, 100), btn_speed_down, border_radius=4)
        minus_txt = font_med.render("Slower", True, (255, 255, 255))
        screen.blit(minus_txt, (btn_speed_down.x + 8, btn_speed_down.y + 5))

        pygame.draw.rect(screen, (100, 100, 100), btn_speed_up, border_radius=4)
        plus_txt = font_med.render("Faster", True, (255, 255, 255))
        screen.blit(plus_txt, (btn_speed_up.x + 8, btn_speed_up.y + 5))

    def draw_legend():
        """Draw a small legend at the bottom."""
        y = grid_offset_y + grid_height + 10
        x = grid_offset_x
        items = [
            (COLORS[ROAD], "Road"),
            (COLORS[BUILDING], "Building"),
            (COLORS[DELIVERY], "Delivery"),
            (COLORS[TRAFFIC], "Traffic"),
            (COLORS[BASE_STATION], "Base"),
            (COLORS['robot'], "Robot"),
            (COLORS['path'], "Path"),
        ]
        for color, name in items:
            pygame.draw.rect(screen, color, (x, y, 14, 14))
            pygame.draw.rect(screen, (100, 100, 100), (x, y, 14, 14), 1)
            label = font_small.render(name, True, COLORS['text'])
            screen.blit(label, (x + 18, y))
            x += 90

    # Main loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos

                # Algorithm selection
                for rect, idx in algo_buttons:
                    if rect.collidepoint(mx, my) and not sim_running:
                        selected_algo = idx

                # Start button
                if btn_start.collidepoint(mx, my) and not sim_running:
                    if remaining_deliveries:
                        start_next_delivery()

                # Reset button
                if btn_reset.collidepoint(mx, my):
                    reset_simulation()

                # Speed buttons
                if btn_speed_up.collidepoint(mx, my):
                    animation_speed = max(1, animation_speed - 2)
                if btn_speed_down.collidepoint(mx, my):
                    animation_speed = min(30, animation_speed + 2)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    sim_paused = not sim_paused
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # Update simulation
        if sim_running and not sim_paused:
            frame_counter += 1
            if frame_counter >= animation_speed:
                frame_counter = 0
                if path_index < len(current_path) - 1:
                    path_index += 1
                    robot_pos = current_path[path_index]
                    if robot_pos not in visited_trail:
                        visited_trail.append(robot_pos)
                else:
                    # reached destination
                    goal = remaining_deliveries.pop(0)
                    completed_deliveries.append(goal)
                    current_delivery_idx += 1
                    sim_running = False

                    # automatically start next if there are more
                    if remaining_deliveries:
                        start_next_delivery()

        # Draw everything
        screen.fill(COLORS['bg'])

        # Title bar
        title_text = font_title.render(
            "MODULE 1 - Intelligent Urban Delivery Robot", True, COLORS['text'])
        screen.blit(title_text, (grid_offset_x, 15))

        draw_grid_cells()
        draw_path_preview()
        draw_robot()
        draw_panel()
        draw_legend()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# PERFORMANCE COMPARISON (CONSOLE)

def run_performance_comparison(grid, cost_grid, base, deliveries):
    """
    Run all five algorithms on the same delivery sequence
    and print a comparison table.
    """
    print("\n" + "=" * 75)
    print("           PERFORMANCE COMPARISON ACROSS ALL ALGORITHMS")
    print("=" * 75)

    all_results = {}

    for algo_name, algo_func in ALGORITHMS.items():
        print(f"\n--- {algo_name} ---")
        current_pos = base
        total_cost = 0
        total_nodes = 0
        total_time = 0.0
        delivery_details = []

        for i, goal in enumerate(deliveries):
            t0 = time.perf_counter()
            path, cost, nodes = algo_func(grid, cost_grid, current_pos, goal)
            t1 = time.perf_counter()
            elapsed = t1 - t0

            if path:
                print(f"  Delivery {i+1} to {goal}: cost={cost}, "
                      f"nodes={nodes}, time={elapsed:.5f}s, "
                      f"path length={len(path)}")
                total_cost += cost
                total_nodes += nodes
                total_time += elapsed
                current_pos = goal
                delivery_details.append({
                    'delivery': i + 1,
                    'cost': cost,
                    'nodes': nodes,
                    'time': elapsed,
                    'path_len': len(path),
                })
            else:
                print(f"  Delivery {i+1} to {goal}: NO PATH FOUND")
                delivery_details.append({
                    'delivery': i + 1,
                    'cost': float('inf'),
                    'nodes': nodes,
                    'time': elapsed,
                    'path_len': 0,
                })

        all_results[algo_name] = {
            'total_cost': total_cost,
            'total_nodes': total_nodes,
            'total_time': total_time,
            'details': delivery_details,
        }

        print(f"  TOTALS: cost={total_cost}, nodes={total_nodes}, "
              f"time={total_time:.5f}s")

    # Summary table
    print("\n" + "=" * 75)
    print(f"{'Algorithm':<22} {'Total Cost':>12} {'Nodes Explored':>16} "
          f"{'Time (s)':>12}")
    print("-" * 75)
    for algo_name in ALGORITHMS:
        r = all_results[algo_name]
        print(f"{algo_name:<22} {r['total_cost']:>12} "
              f"{r['total_nodes']:>16} {r['total_time']:>12.5f}")
    print("=" * 75)

    # Determine best algorithm
    best_cost_algo = min(all_results, key=lambda x: all_results[x]['total_cost'])
    best_time_algo = min(all_results, key=lambda x: all_results[x]['total_time'])
    best_nodes_algo = min(all_results, key=lambda x: all_results[x]['total_nodes'])

    print(f"\nBest by Path Optimality (lowest cost): {best_cost_algo} "
          f"(cost = {all_results[best_cost_algo]['total_cost']})")
    print(f"Best by Execution Time: {best_time_algo} "
          f"(time = {all_results[best_time_algo]['total_time']:.5f}s)")
    print(f"Best by Search Efficiency (fewest nodes): {best_nodes_algo} "
          f"(nodes = {all_results[best_nodes_algo]['total_nodes']})")

    return all_results


# Main

def main():
    print("=" * 55)
    print("  MODULE 1 - Intelligent Urban Delivery Robot")
    print("=" * 55)

    # Use a fixed seed so results are reproducible
    seed = 42
    grid, cost_grid, base, deliveries = create_environment(seed=seed)

    print(f"\nEnvironment created (15x15 grid)")
    print(f"  Base station at: {base}")
    print(f"  Delivery locations: {deliveries}")
    print()

    # Show the grid in console
    print_grid(grid, cost_grid, robot_pos=base)

    # Run the full performance comparison in the console
    results = run_performance_comparison(grid, cost_grid, base, deliveries)

    # Launch graphical visualization if pygame is available
    if PYGAME_AVAILABLE:
        print("\nLaunching graphical visualization...")
        print("Controls:")
        print("  - Click an algorithm to select it")
        print("  - Click 'Start Delivery' to begin")
        print("  - SPACE to pause/resume")
        print("  - ESC to quit")
        print("  - Use Faster/Slower buttons to adjust speed")
        run_pygame_visualization(grid, cost_grid, base, deliveries)
    else:
        print("\n(Install pygame for graphical visualization: pip install pygame)")


if __name__ == "__main__":
    main()
