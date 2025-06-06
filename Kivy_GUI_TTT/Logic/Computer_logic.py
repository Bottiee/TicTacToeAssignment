import random

class ComputerPlayer:
    def __init__(self, symbol):
        self.symbol = symbol  # 'X' or 'O'

    def opponent_symbol(self):
        return 'O' if self.symbol == 'X' else 'X'

    def find_winning_move(self, grid):
        size = len(grid)
        lines = []

        # Collect all rows, cols, diagonals as (line_cells, coordinates)
        # Each line is a list of (cell_value, (row, col))
        # for row_idx in range(size):
        #     line = [(grid[row_idx][col], (row_idx, col)) for col in range(size)]
        #     lines.append(line)
        #
        # for col_idx in range(size):
        #     line = [(grid[row][col_idx], (row, col_idx)) for row in range(size)]
        #     lines.append(line)
        #
        # main_diag = [(grid[index][index], (index, index)) for index in range(size)]
        # anti_diag = [(grid[index][size - 1 - index], (index, size - 1 - index)) for index in range(size)]
        # lines.append(main_diag)
        # lines.append(anti_diag)

        # Check each line for winning/blocking opportunity
        # for line in lines:
        #     symbols = [cell for cell, _ in line]
        #     if symbols.count(self.symbol) == size - 1 and symbols.count(' ') == 1:
        #         # Return the empty spot to win
        #         for cell, coord in line:
        #             if cell == ' ':
        #                 return coord
        #
        # return None

    def find_blocking_move(self, grid):
        size = len(grid)
        opponent = self.opponent_symbol()
        lines = []

        for row_idx in range(size):
            line = [(grid[row_idx][col], (row_idx, col)) for col in range(size)]
            lines.append(line)

        for col_idx in range(size):
            line = [(grid[row][col_idx], (row, col_idx)) for row in range(size)]
            lines.append(line)

        main_diag = [(grid[index][index], (index, index)) for index in range(size)]
        anti_diag = [(grid[index][size - 1 - index], (index, size - 1 - index)) for index in range(size)]
        lines.append(main_diag)
        lines.append(anti_diag)

        for line in lines:
            symbols = [cell for cell, _ in line]
            if symbols.count(opponent) == size - 1 and symbols.count(' ') == 1:
                # Return the empty spot to block opponent win
                for cell, coord in line:
                    if cell == ' ':
                        return coord
        return None

    @staticmethod
    def choose_random_move(grid):
        size = len(grid)
        empty_cells = [(row, col) for row in range(size) for col in range(size) if grid[row][col] == ' ']
        return random.choice(empty_cells) if empty_cells else None

    # def make_move(self, grid):
    #     # Try to win
    #     move = self.find_winning_move(grid)
    #     if move:
    #         return move

        # Try to block opponent
        # move = self.find_blocking_move(grid)
        # if move:
        #     return move

        # Otherwise pick random
        # return self.choose_random_move(grid)

def get_cpu_move(grid):
    size = len(grid)
    empty_cells = [(rows, columns) for rows in range(size) for columns in range(size) if grid[rows][columns] == '']
    return random.choice(empty_cells) if empty_cells else None