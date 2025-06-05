# Grid_logic.py

# class Grid:
#     def __init__(self, size):
#         if size not in [3, 4, 5]:
#             raise ValueError("Unsupported grid size. Choose 3, 4, or 5.")
#         self.size = size
#         self.grid = [[' ' for _ in range(size)] for _ in range(size)]
#
#     def get_rows(self):
#         return self.grid
#
#     def get_columns(self):
#         return [[self.grid[row][col] for row in range(self.size)] for col in range(self.size)]
#
#     def get_diagonals(self):
#         main_diag = [self.grid[index][index] for index in range(self.size)]
#         anti_diag = [self.grid[index][self.size - index - 1] for index in range(self.size)]
#         return [main_diag, anti_diag]
#
#     def display(self):
#         dash_line = '-' * (self.size * 4 - 3)
#         print(dash_line)
#         for row in self.grid:
#             print(' | '.join(row))
#             print(dash_line)
