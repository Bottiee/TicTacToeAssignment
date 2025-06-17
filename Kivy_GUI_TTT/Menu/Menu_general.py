from Kivy_GUI_TTT.Game_history.History_storage import HistoryStorage

class MenuManager:
    def __init__(self, screen_manager):
        self.screen_manager = screen_manager
        self.grid_size = 3
        self.history_storage = HistoryStorage()
        self.cpu_enabled = True


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
        if size in (3, 4, 5, 6, 7, 8, 9):
            self.grid_size = size
        else:
            raise ValueError("Invalid grid size. Must be 3 to 9.")

    def get_tile_size(self) -> int:
        return self.grid_size

    def close(self):
        self.history_storage.close()
