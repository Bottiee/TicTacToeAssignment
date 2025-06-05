# User_input_checks.py
#
# def parse_input(user_input, grid_size):
#     """
#     Parses a string like '2-2' and returns (row, column) if valid.
#     Otherwise, returns None.
#     """
#     try:
#         if '-' not in user_input:
#             return None
#
#         col_str, row_str = user_input.split('-')
#         column = int(col_str.strip()) - 1
#         row = int(row_str.strip()) - 1
#
#         if 0 <= row < grid_size and 0 <= column < grid_size:
#             return row, column
#         else:
#             return None
#
#     except ValueError:
#         return None
