from Inputs_and_prompts.Prompt_dictionary import Prompts
from Logic.Grid_logic import Grid
from Inputs_and_prompts.User_input_checks import parse_input
from Logic.Conditionals import check_winner, check_tie, handle_game_end
from Logic.Size_logic import SizeLogic
from Logic.Computer_logic import ComputerPlayer
from Inputs_and_prompts.Menu_general import MenuManager

prompts = Prompts()
game_results = []
menu_manager = MenuManager()
size_logic = SizeLogic()

def main_menu():
    while True:
        print(prompts.main_menu_text)
        user_input = input(prompts.main_menu_input)

        if user_input == '1':
            start_game()
            return
        elif user_input == '2':
            options_menu()
            return
        elif user_input == '3':
            print(prompts.exit_game)
            return
        else:
            print(prompts.invalid_input)

def options_menu():
    print(prompts.options_menu_text)
    print(prompts.clear_history_option)

    user_input = input(prompts.options_menu_input)

    if user_input == '1':
        print(menu_manager.get_history_summary())
        options_menu()
        return
    elif user_input == '2':
        change_grid_size_menu()
        options_menu()
        return
    elif user_input == '3':
        main_menu()
        return
    elif user_input == '4':
        menu_manager.clear_history()
        print(prompts.history_cleared)
        options_menu()
        return
    else:
        print(prompts.invalid_input)
        options_menu()
        return

def change_grid_size_menu():
    print(prompts.grid_size_prompt)
    for size in size_logic.available_sizes:
        print(f"{size}x{size}")
    user_input = input(prompts.grid_size_input)

    try:
        size = int(user_input)
        size_logic.set_size(size)
        print(prompts.grid_size_set.format(size=size))
    except ValueError:
        print(prompts.invalid_grid_size)
    return

def start_game():
    vs_cpu_input = input(prompts.cpu_prompt).strip().lower()
    vs_cpu = vs_cpu_input == 'y'
    menu_manager.set_cpu_enabled(vs_cpu)

    grid = Grid(menu_manager.get_current_grid_size())
    current_player = 'X'
    cpu_player = None

    if vs_cpu:
        cpu_player = ComputerPlayer('O')

    while True:
        grid.display()
        print(prompts.turn_prompt.format(player=current_player))

        if vs_cpu and current_player == cpu_player.symbol:
            move = cpu_player.make_move(grid.grid)
            if move is None:
                print(prompts.cpu_no_moves)
                handle_game_end(None, menu_manager.game_history)
                break
            row, column = move
            print(prompts.cpu_choice.format(col=column + 1, row=row + 1))
        else:
            try:
                user_input = input(prompts.move_input).strip().lower()
                if user_input == 'q':
                    print(prompts.quit_game)
                    return main_menu()
                coords = parse_input(user_input, grid.size)

                if not coords:
                    print(prompts.invalid_move_format)
                    continue

                row, column = coords

                if grid.grid[row][column] != ' ':
                    print(prompts.cell_taken)
                    continue

            except ValueError:
                print(prompts.invalid_numeric_input)
                continue
            except Exception as e:
                print(prompts.unexpected_error.format(error=e))
                continue

        grid.grid[row][column] = current_player

        if check_winner(grid, current_player):
            grid.display()
            menu_manager.record_game_result(current_player)
            handle_game_end(current_player, menu_manager.game_history)
            return main_menu()

        if check_tie(grid):
            grid.display()
            menu_manager.record_game_result(None)
            handle_game_end(None, menu_manager.game_history)
            return main_menu()

        current_player = 'O' if current_player == 'X' else 'X'
    return main_menu()

if __name__ == "__main__":
    main_menu()
