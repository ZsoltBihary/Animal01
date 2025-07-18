import pygame
import sys
import random

# Grid and game settings
GRID_SIZE = 10
CELL_SIZE = 50
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 5  # Controls how fast the roach moves

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)     # Food
RED = (200, 0, 0)       # Poison
BLUE = (50, 50, 255)    # Roach

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Roach Grid World")
clock = pygame.time.Clock()

# Game state
roach_pos = [0, 0]
food_positions = [[2, 3], [6, 6]]
poison_positions = [[4, 4], [7, 2]]


def draw_grid():
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (WINDOW_SIZE, y))


def draw_entity(pos, color, shape="rect"):
    x, y = pos[0] * CELL_SIZE, pos[1] * CELL_SIZE
    if shape == "circle":
        pygame.draw.circle(screen, color, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 3)
    else:
        pygame.draw.rect(screen, color, (x + 5, y + 5, CELL_SIZE - 10, CELL_SIZE - 10))


def move_roach():
    # Dummy policy: move randomly
    direction = random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
    new_x = max(0, min(GRID_SIZE - 1, roach_pos[0] + direction[0]))
    new_y = max(0, min(GRID_SIZE - 1, roach_pos[1] + direction[1]))
    roach_pos[0], roach_pos[1] = new_x, new_y


# Main loop
running = True
while running:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_grid()

    # Draw food
    for food in food_positions:
        draw_entity(food, GREEN)

    # Draw poison
    for poison in poison_positions:
        draw_entity(poison, RED)

    # Move and draw roach
    move_roach()
    draw_entity(roach_pos, BLUE, shape="circle")

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()


# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
#
# # Configuration
# grid_size = 10
# roach_pos = [5, 5]  # Starting position
# food_positions = [[2, 3], [7, 7]]
# poison_positions = [[4, 4], [8, 1]]
#
# # Initialize figure
# fig, ax = plt.subplots()
# im = ax.imshow(np.zeros((grid_size, grid_size)), cmap="gray_r", vmin=0, vmax=3)
#
#
# def update(frame):
#     global roach_pos
#
#     # Dummy policy for movement: move right then down
#     if roach_pos[0] < grid_size - 1:
#         roach_pos[0] += 1
#     elif roach_pos[1] < grid_size - 1:
#         roach_pos[1] += 1
#
#     # Clear grid
#     grid = np.zeros((grid_size, grid_size))
#
#     # Add food and poison
#     for f in food_positions:
#         grid[f[0], f[1]] = 1  # Food
#     for p in poison_positions:
#         grid[p[0], p[1]] = 2  # Poison
#
#     # Add roach
#     grid[roach_pos[0], roach_pos[1]] = 3  # Roach
#
#     im.set_array(grid)
#     return [im]
#
#
# ani = animation.FuncAnimation(fig, update, frames=50, interval=200, blit=True)
#
# # To view live
# plt.show()
#
# # To save
# # ani.save("roach_behavior.mp4", fps=5)
