# Conditionals.py

def check_win_condition(game_board, player_symbol):
    board_size = len(game_board)

    # Check rows for a winning sequence of three
    for row_index in range(board_size):
        for column_index in range(board_size - 2):
            if (game_board[row_index][column_index] == player_symbol and
                game_board[row_index][column_index + 1] == player_symbol and
                game_board[row_index][column_index + 2] == player_symbol):
                return True

    # Check columns for a winning sequence of three
    for column_index in range(board_size):
        for row_index in range(board_size - 2):
            if (game_board[row_index][column_index] == player_symbol and
                game_board[row_index + 1][column_index] == player_symbol and
                game_board[row_index + 2][column_index] == player_symbol):
                return True

    # Check main diagonal (top-left to bottom-right) for a winning sequence of three
    for row_index in range(board_size - 2):
        for column_index in range(board_size - 2):
            if (game_board[row_index][column_index] == player_symbol and
                game_board[row_index + 1][column_index + 1] == player_symbol and
                game_board[row_index + 2][column_index + 2] == player_symbol):
                return True

    # Check anti-diagonal (top-right to bottom-left) for a winning sequence of three
    for row_index in range(board_size - 2):
        for column_index in range(2, board_size):
            if (game_board[row_index][column_index] == player_symbol and
                game_board[row_index + 1][column_index - 1] == player_symbol and
                game_board[row_index + 2][column_index - 2] == player_symbol):
                return True

    return False


def check_draw_condition(game_board):
    return all(cell != '' for row in game_board for cell in row)


### OLD CODE ###

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