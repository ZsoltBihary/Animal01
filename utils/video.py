import pygame
import sys
import random
import imageio
import numpy as np

# --- Settings ---
GRID_SIZE = 10
CELL_SIZE = 50
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 5
TOTAL_FRAMES = 50  # length of the video

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)     # Food
RED = (200, 0, 0)       # Poison
BLUE = (50, 50, 255)    # Roach

# --- Game State ---
roach_pos = [0, 0]
food_positions = [[2, 3], [6, 6]]
poison_positions = [[4, 4], [7, 2]]

# --- Initialize Pygame ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Roach Grid World")
clock = pygame.time.Clock()

# --- Frame storage for video ---
frames = []


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
    # Random move (replace with DQN logic in future)
    dx, dy = random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
    new_x = max(0, min(GRID_SIZE - 1, roach_pos[0] + dx))
    new_y = max(0, min(GRID_SIZE - 1, roach_pos[1] + dy))
    roach_pos[0], roach_pos[1] = new_x, new_y


# --- Main Loop ---
for frame in range(TOTAL_FRAMES):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(WHITE)
    draw_grid()

    # Draw food and poison
    for food in food_positions:
        draw_entity(food, GREEN)
    for poison in poison_positions:
        draw_entity(poison, RED)

    # Move roach and draw it
    move_roach()
    draw_entity(roach_pos, BLUE, shape="circle")

    pygame.display.flip()
    clock.tick(FPS)

    # Capture frame
    frame_data = pygame.surfarray.array3d(screen)
    frame_data = np.transpose(frame_data, (1, 0, 2))  # From (width, height, channels) to (height, width, channels)
    frames.append(frame_data)

# --- Save video or GIF ---
# Save as GIF
imageio.mimsave("roach_sim.gif", frames, fps=FPS)

# Optional: Save as MP4 (lower quality)
# imageio.mimsave("roach_sim.mp4", frames, fps=FPS)

print("✅ Animation saved as 'roach_sim.gif'")
pygame.quit()
