import pygame
import torch
import numpy as np
import imageio
from utils.helper import EMPTY, FOOD, POISON, ANIMAL
import imageio.v2 as imageio2  # At the top if not already imported
# import os


class GridRenderer:
    # Colors
    GREY = (200, 200, 200)  # EMPTY, fog of war
    WHITE = (255, 255, 255)  # EMPTY, visible
    BLACK = (0, 0, 0)  # Grid lines
    GREEN = (0, 200, 0)  # FOOD
    RED = (200, 0, 0)  # POISON
    BLUE = (50, 50, 255)  # ANIMAL

    def __init__(self, H: int, W: int, cell_size=32):
        pygame.init()
        self.H = H
        self.W = W
        self.cell_size = cell_size
        self.screen = None

    def _init_screen(self, visible_mask):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.W * self.cell_size, self.H * self.cell_size))
            pygame.display.set_caption("Grid Visualization")
        self.screen.fill(self.GREY)
        # Apply visible_mask to shade visible region
        for pos_y in range(self.H):
            for pos_x in range(self.W):
                x, y = pos_x * self.cell_size, pos_y * self.cell_size
                if visible_mask[pos_y, pos_x]:
                    pygame.draw.rect(self.screen, self.WHITE, (x, y, self.cell_size, self.cell_size))
        self.draw_grid_lines()

    def draw_grid_lines(self):
        for x in range(0, self.W * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.BLACK, (x, 0), (x, self.H * self.cell_size))
        for y in range(0, self.H * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.BLACK, (0, y), (self.W * self.cell_size, y))

    def draw_grid(self, snapshot):
        """Draw a grid: tensor of shape (H, W)"""
        grid = snapshot[0]
        visible_mask = snapshot[1]
        H, W = grid.shape
        self._init_screen(visible_mask)
        for pos_y in range(H):
            for pos_x in range(W):
                cell = int(grid[pos_y, pos_x])
                x, y = pos_x * self.cell_size, pos_y * self.cell_size
                if cell == ANIMAL:
                    pygame.draw.circle(self.screen, self.BLUE,
                                       (x + self.cell_size // 2, y + self.cell_size // 2), self.cell_size // 3)
                elif cell == FOOD:
                    pygame.draw.circle(self.screen, self.GREEN,
                                       (x + self.cell_size // 2, y + self.cell_size // 2), self.cell_size // 4)
                elif cell == POISON:
                    pygame.draw.rect(self.screen, self.RED,
                                     (x + self.cell_size // 4, y + self.cell_size // 4,
                                      self.cell_size // 2, self.cell_size // 2))
                else:
                    pass
                    # pygame.draw.rect(self.screen, self.WHITE,
                    #                  (x + 2, y + 2, self.cell_size - 4, self.cell_size - 4))

        pygame.display.flip()

    def get_surface_image(self):
        """Return current screen as a NumPy image (H_px, W_px, 3)"""
        raw = pygame.surfarray.array3d(self.screen)
        return np.transpose(raw, (1, 0, 2))  # (H, W, 3)

    def play_episode(self, history, delay_ms=200):
        """Display each frame in a grid_episode: tensor of shape (T, H, W)"""
        for snapshot in history:
            self.handle_events()
            self.draw_grid(snapshot)
            pygame.time.delay(delay_ms)

    def handle_events(self):
        """Handle basic quit events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    # def save_episode_as_gif(self, grid_episode, gif_path, delay_ms=500, slow_factor=10):
    #     """Save episode to GIF, slowing it down by repeating frames"""
    #     frames = []
    #     for frame in grid_episode:
    #         self.draw_grid(frame)
    #         img = self.get_surface_image()
    #         # Repeat the same image slow_factor times
    #         for _ in range(slow_factor):
    #             frames.append(img.copy())  # ensure distinct copies
    #         pygame.time.delay(delay_ms // 10)  # just for smoother preview
    #
    #     imageio.mimsave(gif_path, frames, duration=delay_ms / 1000.0)

    def save_episode_as_mp4(self, history, video_path, fps=1):
        """Save episode as MP4 using imageio and ffmpeg backend"""
        frames = []
        for snapshot in history:
            self.draw_grid(snapshot)
            img = self.get_surface_image()
            frames.append(img)
        imageio2.mimwrite(video_path, frames, fps=fps, codec='libx264')


# ------------------ SANITY CHECK ------------------ #
if __name__ == "__main__":
    def create_dummy_episode(T=10, H=5, W=7):
        grid_episode = torch.zeros((T, H, W), dtype=torch.uint8)
        for t in range(T):
            grid_episode[t, 1, 1] = FOOD
            grid_episode[t, 2, 2] = POISON
            if t < min(H, W):
                grid_episode[t, t, t] = ANIMAL
        return grid_episode


    renderer = GridRenderer(H=5, W=7, cell_size=64)
    demo_episode = create_dummy_episode()

    # Display animation
    renderer.play_episode(demo_episode, delay_ms=300)

    # Save as GIF
    output_gif_path = "demo_episode.gif"
    renderer.save_episode_as_gif(demo_episode, gif_path=output_gif_path, delay_ms=200, slow_factor=1)

    # renderer.save_episode_as_mp4(grid_episode=demo_episode, video_path, fps=1)
    renderer.save_episode_as_mp4(demo_episode, video_path="demo_episode.mp4", fps=4)

    # output_gif_path
