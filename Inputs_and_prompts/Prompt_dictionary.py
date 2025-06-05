# Prompt_dictionary.py

class Prompts:
    def __init__(self):
        # === Main Menu ===
        self.main_menu_text = "=== Main Menu ===\n1. Start Game\n2. Options\n3. Quit"
        self.main_menu_input = "Select an option: "
        self.exit_game = "Quitting the game. Goodbye!"
        self.invalid_input = "Invalid input. Please try again."

        # === Options Menu ===
        self.options_menu_text = "=== Options Menu ===\n1. View Game History\n2. Change Tile Size\n3. Return to Main Menu"
        self.options_menu_input = "Select an option: "
        self.clear_history_option = "4. Clear History"
        self.history_cleared = "Game history cleared successfully."

        # === Grid Size ===
        self.grid_size_prompt = "Select grid size:"
        self.grid_size_input = "Enter your choice: "
        self.grid_size_set = "Grid size set to {size}x{size}"
        self.invalid_grid_size = "Invalid input or unsupported size."

        # === Game Start / Turn Handling ===
        self.cpu_prompt = "Play against CPU? (y/n): "
        self.turn_prompt = "Player {player}'s turn."
        self.cpu_no_moves = "No moves left! It's a tie."
        self.cpu_choice = "CPU chooses: {col}-{row}"

        # === Move Input ===
        self.move_input = "Enter your move as Column-Row (e.g., 2-2), or 'q' to quit: "
        self.quit_game = "Quitting the game..."
        self.invalid_move_format = "Invalid input format or out-of-bounds. Please use 'Column-Row' within grid limits."
        self.cell_taken = "Cell already taken. Try another."
        self.invalid_numeric_input = "Invalid input. Please enter numeric values."
        self.unexpected_error = "Unexpected error: {error}"

        # === Unused Stubs (keep for dev/testing purposes) ===
        self.start_game = "Starting game...\n(Game logic not yet implemented)"
        self.history_stub = "Displaying game history...\n"
        self.tileset_stub = "Changing tile size...\n(Tile size logic not yet implemented)"
