# Conditionals.py

# def check_winner(grid_obj, player):
#     # Check rows
#     for row in grid_obj.get_rows():
#         if all(cell == player for cell in row):
#             return True
#
#     # Check columns
#     for col in grid_obj.get_columns():
#         if all(cell == player for cell in col):
#             return True
#
#     # Check diagonals
#     for diag in grid_obj.get_diagonals():
#         if all(cell == player for cell in diag):
#             return True
#
#     return False

# def check_tie(grid_obj):
#     return all(cell != ' ' for row in grid_obj.get_rows() for cell in row)

    return False

# def handle_game_end(winner, game_results):
#     if winner:
#         print(f"Player {winner} wins!")
#     else:
#         print("It's a tie!")
#
#     # Append results to game history (not yet implemented)
#     game_results.append(winner or "Tie")
#
#     input("Press Enter to return to the main menu...")


def check_win_condition(board, symbol):
    size = len(board)

    # Check rows and columns
    for index in range(size):
        if all(cell == symbol for cell in board[index]):
            return True
        if all(row[index] == symbol for row in board):
            return True

    # Check diagonals
    if all(board[index][index] == symbol for index in range(size)):
        return True
    if all(board[index][size - 1 - index] == symbol for index in range(size)):
        return True

    return False


def check_draw_condition(board):
    return all(cell != '' for row in board for cell in row)
