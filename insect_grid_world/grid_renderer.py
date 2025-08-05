import pygame
import torch
import numpy as np
import imageio.v2 as imageio
from grid_world import GridWorld

# EMPTY, FOOD, POISON = 0, 1, 2
EMPTY, FOOD, BAR1, BAR2, BAR3, BAR4, BAR5, ANIMAL = 0, 1, 2, 3, 4, 5, 6, 7


class GridRenderer:
    # Colors
    INVISIBLE = (0, 0, 0)
    VISIBLE = (15, 15, 15)
    LINE = (31, 31, 31)
    # EMPTY, FOOD, BAR1, BAR2, BAR3, BAR4, BAR5, ANIMAL = 0, 1, 2, 3, 4, 5, 6, 7
    COLOR = [VISIBLE,  # EMPTY
             (0, 255, 15),  # FOOD
             (55, 40, 0),  # BAR1
             (105, 30, 0),  # BAR2
             (155, 10, 0),  # BAR3
             (205, 10, 0),  # BAR4
             (255, 0, 0),  # BAR5
             (127, 127, 255)
             ]
    # COLOR = [VISIBLE,  # EMPTY
    #          (0, 255, 15),  # FOOD
    #          (67, 60, 0),   # BAR1
    #          (111, 45, 0),  # BAR2
    #          (159, 30, 0),  # BAR3
    #          (207, 15, 0),  # BAR4
    #          (255, 0, 0),  # BAR5
    #          (127, 127, 255)
    #          ]

    def __init__(self, world: GridWorld, text_info: dict, cell_size=63, egocentric=False):
        pygame.init()

        self.H = world.H
        self.W = world.W
        self.R = world.R
        self.text_info = text_info
        self.cell_size = cell_size
        self.egocentric = egocentric

        self.margin = cell_size
        self.draw_width = self.W * self.cell_size
        self.draw_height = self.H * self.cell_size
        self.total_width = round_up_16(self.draw_width + 2 * self.margin)
        self.total_height = round_up_16(self.draw_height + 2 * self.margin + 3 * self.cell_size)
        # We want some blank space in the bottom
        self.screen = None

    def draw_snapshot(self, grid: torch.Tensor, animal_pos: torch.Tensor, egocentric: bool = None):
        if egocentric is None:
            egocentric = self.egocentric

        if egocentric:
            self.draw_egocentric(grid, animal_pos)
        else:
            self.draw(grid, animal_pos)

    def init_screen(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.total_width, self.total_height))
            pygame.display.set_caption("Grid Simulation")
        self.screen.fill(self.INVISIBLE)
        # self.draw_grid_lines()

    def shade_visibility(self, animal_pos: torch.Tensor, radius: int = 0):
        if radius == 0:
            radius = self.R
        offset = self.margin
        center_y, center_x = animal_pos.tolist()

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y = (center_y + dy) % self.H
                x = (center_x + dx) % self.W

                rect = pygame.Rect(
                    offset + x * self.cell_size,
                    offset + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, self.VISIBLE, rect)

    def draw_grid_lines(self):
        offset = self.margin
        for x in range(self.W + 1):
            x_pos = offset + x * self.cell_size
            pygame.draw.line(self.screen, self.LINE, width=3,
                             start_pos=(x_pos, offset), end_pos=(x_pos, offset + self.draw_height))
        for y in range(self.H + 1):
            y_pos = offset + y * self.cell_size
            pygame.draw.line(self.screen, self.LINE, width=3,
                             start_pos=(offset, y_pos), end_pos=(offset + self.draw_width, y_pos))

    def draw(self, grid: torch.Tensor, animal_pos: torch.Tensor):
        self.init_screen()
        self.shade_visibility(animal_pos=animal_pos, radius=0)
        self.draw_grid_lines()
        grid = grid.cpu().numpy()
        pos_y, pos_x = animal_pos.tolist()

        for y in range(self.H):
            for x in range(self.W):
                val = grid[y, x].item()
                col = self.COLOR[val]
                rect = pygame.Rect(
                    self.margin + x * self.cell_size,
                    self.margin + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                if val == FOOD:
                    pygame.draw.circle(self.screen, col, rect.center, self.cell_size // 4)
                elif (val >= BAR1) & (val <= BAR5):
                    pygame.draw.rect(self.screen, col,
                                     rect.inflate(2*val-8-2*self.cell_size // 3, 2*val-8-2*self.cell_size // 3))

        # Draw agent
        center_x = self.margin + (pos_x + 0.5) * self.cell_size
        center_y = self.margin + (pos_y + 0.5) * self.cell_size
        pygame.draw.circle(self.screen, self.COLOR[ANIMAL], (int(center_x), int(center_y)), self.cell_size // 3)

        val = grid[pos_y, pos_x].item()
        if val < ANIMAL:
            col = self.COLOR[val]
        else:
            col = self.VISIBLE
        pygame.draw.circle(self.screen, col, (int(center_x), int(center_y)), self.cell_size // 6)

        # if val == FOOD:
        #     pygame.draw.circle(self.screen, self.GREEN, (int(center_x), int(center_y)), self.cell_size // 6)
        # elif val == POISON:
        #     pygame.draw.circle(self.screen, self.RED, (int(center_x), int(center_y)), self.cell_size // 6)
        # else:
        #     pygame.draw.circle(self.screen, self.VISIBLE, (int(center_x), int(center_y)), self.cell_size // 6)

        # Draw text
        self.draw_text()
        pygame.display.flip()

    def draw_text(self):

        font = pygame.font.Font(None, size=2*self.cell_size // 3)  # Default font, size 36

        # Split the keys into three groups
        left_keys = ["Animal", "Model", "Temperature"]
        center_keys = ["Food_reward", "Max_bar_reward", "Move_reward"]
        right_keys = ["Food_density", "Barrier_density", "Mean_reward"]

        # Starting positions
        left_x = self.margin + self.cell_size // 2
        center_x = left_x + 7 * self.cell_size  # adjust for spacing
        right_x = center_x + 6 * self.cell_size  # adjust for spacing
        start_y = self.draw_height + self.cell_size + self.cell_size // 2
        line_height = self.cell_size

        # Draw left column
        y = start_y
        for key in left_keys:
            if key in self.text_info:
                text_surface = font.render(f"{key}: {self.text_info[key]}", True, (255, 255, 255))
                self.screen.blit(text_surface, (left_x, y))
                y += line_height

        # Draw center column
        y = start_y
        for key in center_keys:
            if key in self.text_info:
                text_surface = font.render(f"{key}: {self.text_info[key]}", True, (255, 255, 255))
                self.screen.blit(text_surface, (center_x, y))
                y += line_height

        # Draw right column
        y = start_y
        for key in right_keys:
            if key in self.text_info:
                text_surface = font.render(f"{key}: {self.text_info[key]}", True, (255, 255, 255))
                self.screen.blit(text_surface, (right_x, y))
                y += line_height

    def draw_egocentric(self, grid: torch.Tensor, animal_pos: torch.Tensor):
        H, W = self.H, self.W
        assert H % 2 == 1 and W % 2 == 1, "Grid dimensions must be odd for egocentric view"

        pos_y, pos_x = animal_pos.tolist()
        center_y, center_x = H // 2, W // 2

        shift_y = center_y - pos_y
        shift_x = center_x - pos_x

        rolled_grid = torch.roll(grid, shifts=(shift_y, shift_x), dims=(0, 1))
        new_animal_pos = torch.tensor([center_y, center_x], dtype=torch.long)

        self.draw(rolled_grid, new_animal_pos)

    def play_simulation(self, result, delay_ms=200):
        for i in range(len(result)):
            self.handle_events()
            data = result[i]
            self.draw_snapshot(data["grid"], data["animal_pos"])
            pygame.time.delay(delay_ms)

    def save_as_video(self, result, video_path, fps=6):
        frames = []
        for i in range(len(result)-1):
            data = result[i]
            data2 = result[i+1]
            self.draw_snapshot(data["grid"], data["animal_pos"])
            img = pygame.surfarray.array3d(self.screen)
            img = np.transpose(img, (1, 0, 2))  # Convert to HWC
            frames.append(img)
            self.draw_snapshot(data["grid"], data2["animal_pos"])
            img = pygame.surfarray.array3d(self.screen)
            img = np.transpose(img, (1, 0, 2))  # Convert to HWC
            frames.append(img)

        imageio.mimwrite(video_path, frames, fps=fps, codec='libx264')

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit


def round_up_16(x):
    return ((x + 15) // 16) * 16
