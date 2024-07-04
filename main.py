import pygame
import math
from queue import PriorityQueue
import random

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Advanced Pathfinding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbours = []
        self.width = width
        self.total_rows = total_rows
        self.weight = 1  # Default weight

    def get_pos(self):
        return self.row, self.col 
    
    def is_closed(self):
        return self.color == RED
    
    def is_open(self):
        return self.color == GREEN
    
    def is_barrier(self):
        return self.color == BLACK
    
    def is_start(self):
        return self.color == ORANGE
    
    def is_end(self):
        return self.color == TURQUOISE
    
    def reset(self):
        self.color = WHITE
        self.weight = 1
    
    def make_start(self):
        self.color = ORANGE
    
    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN
    
    def make_barrier(self):
        self.color = BLACK
    
    def make_end(self):
        self.color = TURQUOISE
    
    def make_path(self):
        self.color = PURPLE
    
    def make_weighted(self, weight):
        self.weight = weight
        self.color = YELLOW  # Visual indication of weighted node
    
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
        if self.weight > 1:
            font = pygame.font.Font(None, 30)
            text = font.render(str(self.weight), True, BLACK)
            win.blit(text, (self.x + self.width // 4, self.y + self.width // 4))

    def update_neighbours(self, grid):
        self.neighbours = []
        # Check all 8 directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            new_row, new_col = self.row + dx, self.col + dy
            if 0 <= new_row < self.total_rows and 0 <= new_col < self.total_rows:
                if not grid[new_row][new_col].is_barrier():
                    self.neighbours.append(grid[new_row][new_col])

    def __lt__(self, other):
        return False

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)  # Euclidean distance

def reconstruct_path(came_from, current, draw):
    path_length = 0
    while current in came_from:
        path_length += current.weight
        current = came_from[current]
        current.make_path()
        draw()
    return path_length

def algorithm(draw, grid, start, end, heuristic):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = heuristic(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            path_length = reconstruct_path(came_from, end, draw)
            end.make_end()
            return True, path_length
        
        for neighbour in current.neighbours:
            temp_g_score = g_score[current] + neighbour.weight

            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + heuristic(neighbour.get_pos(), end.get_pos())
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()
        draw()

        if current != start:
            current.make_closed()

    return False, 0

def make_grid(rows, width):
    grid = []
    gap = width // rows 
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Node(i, j, gap, rows)
            grid[i].append(spot)
    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col

def main(win, width):
    ROWS = 50
    grid = make_grid(ROWS, width)

    start = None
    end = None

    run = True
    started = False
    
    algorithm_type = "A*"  # Default algorithm
    heuristic = h  # Default heuristic

    clock = pygame.time.Clock()

    while run:
        clock.tick(60)  # Limit to 60 FPS
        draw(win, grid, ROWS, width)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if started:
                continue

            if pygame.mouse.get_pressed()[0]:  # LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not start and spot != end:
                    start = spot 
                    start.make_start()
                elif not end and spot != start:
                    end = spot
                    end.make_end()
                elif spot != end and spot != start:
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                if spot == end:
                    end = None
            
            elif pygame.mouse.get_pressed()[1]:  # MIDDLE
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                weight = random.randint(2, 5)
                spot.make_weighted(weight)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)
                    
                    success, path_length = algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end, heuristic)
                    if success:
                        print(f"Path found! Length: {path_length}")
                    else:
                        print("No path found.")
                    started = True

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
                    started = False

                if event.key == pygame.K_m:
                    algorithm_type = "Manhattan" if algorithm_type == "A*" else "A*"
                    heuristic = (lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) if algorithm_type == "Manhattan" else h
                    print(f"Switched to {algorithm_type} algorithm")

                if event.key == pygame.K_g:
                    # Generate a random maze
                    for row in grid:
                        for spot in row:
                            if random.random() < 0.3 and spot != start and spot != end:
                                spot.make_barrier()
                    print("Generated random maze")

    pygame.quit()

if __name__ == "__main__":
    main(WIN, WIDTH)