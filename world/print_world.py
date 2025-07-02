from torch import Tensor
from helper import EMPTY, WALL, FOOD, ANIMAL  # Cell codes
from helper import STAY, UP, DOWN, LEFT, RIGHT  # Animal actions


class PrintWorld:
    def __init__(self):
        self.cell_to_str = {EMPTY: '   ', WALL: ' X ', FOOD: ' o ', ANIMAL: '(A)'}
        self.action_to_str = {STAY: 'STAY', UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}

    def print_board(self, board: Tensor, frame=True):
        H, W = board.shape
        horizontal = " "
        if frame:
            for _ in range(W * 3):
                horizontal = horizontal + "-"
        print(horizontal)
        # Iterate through each row of the board
        for row in board:
            row_chars = [self.cell_to_str[value.item()] for value in row]
            row_string = ''.join(row_chars)
            if frame:
                print("|" + row_string + "|")
            else:
                print(row_string)
        print(horizontal)


# ------------------ SANITY CHECK ------------------ #
if __name__ == "__main__":
    from grid_world import GridWorld

    H, W, B = 7, 11, 3
    env = GridWorld(height=H, width=W, batch_size=B, wall_density=0.15, food_density=0.05,
                    device='cpu')
    printer = PrintWorld()
    frame = True
    printer.print_board(env.board[0, ...], frame=frame)
    printer.print_board(env.board[1, ...], frame=frame)
    printer.print_board(env.board[2, ...], frame=frame)
