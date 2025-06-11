# Menu_general.py
from Kivy_GUI_TTT.Game_history.History_storage import HistoryStorage

class MenuManager:
    def __init__(self):
        self.grid_size = 3
        self.history_storage = HistoryStorage()
        self.cpu_enabled = True  # Flag to track CPU mode

    def is_cpu_enabled(self):
        return self.cpu_enabled

    def set_cpu_enabled(self, value: bool):
        self.cpu_enabled = value

    def record_game_result(self, winner):
        if winner is None:
            self.history_storage.record_tie()
        elif winner == 'O' and self.is_cpu_enabled():
            self.history_storage.record_win('CPU')
        elif winner == 'X':
            self.history_storage.record_win('X')
        else:
            self.history_storage.record_win('O')

    def get_history_summary(self):
        return self.history_storage.get_summary()

    def get_history_dict(self):
        return {
            "Ties": self.history_storage.ties,
            "Player 1 wins": self.history_storage.player1_wins,
            "Player 2 wins": self.history_storage.player2_wins,
            "CPU wins": self.history_storage.cpu_wins,
            "Total games": self.history_storage.total_games,
        }

    def clear_history(self):
        self.history_storage.clear_history()

    def record_draw(self):
        self.history_storage.record_tie()

    def set_tile_size(self, size: int):
        if size in (3, 4, 5):
            self.grid_size = size
        else:
            raise ValueError("Invalid grid size. Must be 3, 4, or 5.")

    def get_tile_size(self) -> int:
        return self.grid_size

    def close(self):
        self.history_storage.close()


    ### OLD CODE ###

    # def view_history(self):
    #     if not self.game_history:
    #         print("No games played yet.")
    #     else:
    #         print("Game History:")
    #         for index, result in enumerate(self.game_history, start=1):
    #             print(f"{index}. {result}")
    #     input("Press Enter to return to options...")
    #
    # def show_stats(self):
    #     print("Total Wins:")
    #     for player, count in self.stats.items():
    #         print(f"{player}: {count}")
    #     input("Press Enter to return to options...")
    #
    # def change_grid_size(self):
    #     print("Choose grid size (3-5):")
    #     user_input = input("Enter grid size: ")
    #     if user_input in ['3', '4', '5']:
    #         self.grid_size = int(user_input)
    #         print(f"Grid size set to {self.grid_size}x{self.grid_size}")
    #     else:
    #         print("Invalid choice. Grid size unchanged.")
    #     input("Press Enter to return to options...")
