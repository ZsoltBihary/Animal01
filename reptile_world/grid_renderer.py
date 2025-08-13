import pygame
# import torch
from torch import Tensor
import numpy as np
import imageio.v2 as imageio
from reptile_world. config import Config
from reptile_world.simulator import SimulationResult


class GridRenderer:
    # Colors
    INVISIBLE = (0, 0, 0)
    VISIBLE = (15, 15, 15)
    LINE = (31, 31, 31)

    COLOR = [VISIBLE,  # EMPTY
             (255, 255, 0),  # SEED
             (0, 255, 60),  # PLANT
             (0, 100, 255),  # FRUIT
             (150, 0, 0),  # BARR
             (255, 255, 255)  # ANIMAL
             ]

    def __init__(self, conf: Config, cell_size=48):
        pygame.init()
        self.conf = conf
        # === Consume configuration parameters ===
        self.EMPTY, self.SEED, self.PLANT, self.FRUIT, self.BARR, self.ANIMAL \
            = conf.EMPTY, conf.SEED, conf.PLANT, conf.FRUIT, conf.BARR, conf.ANIMAL

        self.B, self.H, self.W, self.R, self.K \
            = conf.batch_size, conf.grid_height, conf.grid_width, conf.obs_radius, conf.obs_size

        # self.text_info = text_info

        # === Initialize class attributes ===
        self.cell_size = cell_size
        self.margin = cell_size
        self.draw_width = self.W * self.cell_size
        self.draw_height = self.H * self.cell_size
        self.total_width = round_up_16(self.draw_width + 2 * self.margin)
        # We want some blank space in the bottom
        self.total_height = round_up_16(self.draw_height + 2 * self.margin + 3 * self.cell_size)

        self.screen = None

    def init_screen(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.total_width, self.total_height))
            pygame.display.set_caption("Grid Simulation")
        self.screen.fill(self.INVISIBLE)
        # self.draw_grid_lines()

    def shade_visibility(self, animal_pos: Tensor):

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

    def draw_snapshot(self, grid: Tensor, animal_pos: Tensor):
        self.init_screen()
        self.shade_visibility(animal_pos=animal_pos)
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

                # if val == self.SEED:
                #     # Small ellipse instead of rect
                #     ellipse_rect = rect.inflate(-self.cell_size * 6 // 10, -self.cell_size * 7 // 10)
                #     pygame.draw.ellipse(self.screen, col, ellipse_rect)

                if val == self.SEED:
                    # Small ellipse instead of rect, shifted downward
                    ellipse_rect = rect.inflate(-self.cell_size * 3 // 5, -self.cell_size * 3 // 4)
                    ellipse_rect.y += self.cell_size // 10  # shift down by 1/8 of cell size
                    pygame.draw.ellipse(self.screen, col, ellipse_rect)

                elif val == self.PLANT:
                    # Smaller leaf-like triangle (polygon)
                    shrink = self.cell_size * 3 // 10
                    pygame.draw.polygon(
                        self.screen, col, [
                            (rect.centerx, rect.top + shrink),
                            (rect.left + shrink, rect.bottom - shrink),
                            (rect.right - shrink, rect.bottom - shrink)
                        ]
                    )

                elif val == self.FRUIT:
                    # Just circle, no stem
                    pygame.draw.circle(self.screen, col, rect.center, self.cell_size // 4)

                elif val == self.BARR:
                    # Smaller filled square
                    barrier_rect = rect.inflate(-self.cell_size * 7 // 10, -self.cell_size * 7 // 10)
                    pygame.draw.rect(self.screen, col, barrier_rect)

        # Draw agent
        rect = pygame.Rect(
            self.margin + pos_x * self.cell_size,
            self.margin + pos_y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.circle(self.screen, self.COLOR[self.ANIMAL], rect.center, self.cell_size // 3)
        # Diagonal lines (slightly inset)
        inset = self.cell_size // 6  # Adjust for how close you want them to corners
        pygame.draw.line(
            self.screen,
            self.COLOR[self.ANIMAL],
            (rect.left + inset, rect.top + inset),
            (rect.right - inset, rect.bottom - inset),
            5
        )
        pygame.draw.line(
            self.screen,
            self.COLOR[self.ANIMAL],
            (rect.right - inset, rect.top + inset),
            (rect.left + inset, rect.bottom - inset),
            5
        )

        val = grid[pos_y, pos_x].item()
        if val < self.ANIMAL:
            col = self.COLOR[val]
        else:
            col = self.VISIBLE
        pygame.draw.circle(self.screen, col, rect.center, self.cell_size // 5)

        pygame.display.flip()

    def play_simulation(self, result: SimulationResult, delay_ms=200):
        for i in range(len(result)):
            self.handle_events()
            (grid,
             animal_pos,
             last_action,
             reward,
             avg_reward) = result.__getitem__(i)

            self.draw_snapshot(grid, animal_pos)
            pygame.time.delay(delay_ms)

    def save_as_video(self, result: SimulationResult, video_path, fps=2):
        frames = []
        for i in range(len(result)-1):
            (grid, animal_pos, last_action, reward, avg_reward) = result.__getitem__(i)
            (grid2, animal_pos2, last_action2, reward2, avg_reward2) = result.__getitem__(i+1)

            self.draw_snapshot(grid, animal_pos)
            img = pygame.surfarray.array3d(self.screen)
            img = np.transpose(img, (1, 0, 2))  # Convert to HWC
            frames.append(img)

            self.draw_snapshot(grid, animal_pos2)
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

    # def draw_text(self):
    #
    #     font = pygame.font.Font(None, size=2*self.cell_size // 3)  # Default font, size 36
    #
    #     # Split the keys into three groups
    #     left_keys = ["Animal", "Model", "Temperature"]
    #     center_keys = ["Food_reward", "Max_bar_reward", "Move_reward"]
    #     right_keys = ["Food_density", "Barrier_density", "Mean_reward"]
    #
    #     # Starting positions
    #     left_x = self.margin + self.cell_size // 2
    #     center_x = left_x + 7 * self.cell_size  # adjust for spacing
    #     right_x = center_x + 6 * self.cell_size  # adjust for spacing
    #     start_y = self.draw_height + self.cell_size + self.cell_size // 2
    #     line_height = self.cell_size
    #
    #     # Draw left column
    #     y = start_y
    #     for key in left_keys:
    #         if key in self.text_info:
    #             text_surface = font.render(f"{key}: {self.text_info[key]}", True, (255, 255, 255))
    #             self.screen.blit(text_surface, (left_x, y))
    #             y += line_height
    #
    #     # Draw center column
    #     y = start_y
    #     for key in center_keys:
    #         if key in self.text_info:
    #             text_surface = font.render(f"{key}: {self.text_info[key]}", True, (255, 255, 255))
    #             self.screen.blit(text_surface, (center_x, y))
    #             y += line_height
    #
    #     # Draw right column
    #     y = start_y
    #     for key in right_keys:
    #         if key in self.text_info:
    #             text_surface = font.render(f"{key}: {self.text_info[key]}", True, (255, 255, 255))
    #             self.screen.blit(text_surface, (right_x, y))
    #             y += line_height
