import pickle
import os

class HistoryStorage:
    def __init__(self, filename='Game_history/game_history.pkl'):
        self.filename = filename
        # Initialize default stats as attributes
        self.ties = 0
        self.player1_wins = 0
        self.player2_wins = 0
        self.cpu_wins = 0
        self.total_games = 0
        self.load_history()

    def ensure_dir_exists(self):
        directory = os.path.dirname(self.filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def load_history(self):
        if not os.path.exists(self.filename):
            self.save_history()  # Create default file
            return
        try:
            with open(self.filename, 'rb') as f:
                data = pickle.load(f)
                # Load into attributes with fallback defaults
                self.ties = data.get('ties', 0)
                self.player1_wins = data.get('player1_wins', 0)
                self.player2_wins = data.get('player2_wins', 0)
                self.cpu_wins = data.get('cpu_wins', 0)
                self.total_games = data.get('total_games', 0)
        except (EOFError, pickle.PickleError):
            self.reset_history()
            self.save_history()

    def save_history(self):
        self.ensure_dir_exists()
        data = {
            'ties': self.ties,
            'player1_wins': self.player1_wins,
            'player2_wins': self.player2_wins,
            'cpu_wins': self.cpu_wins,
            'total_games': self.total_games,
        }
        with open(self.filename, 'wb') as file:
            pickle.dump(data, file)

    def reset_history(self):
        self.ties = 0
        self.player1_wins = 0
        self.player2_wins = 0
        self.cpu_wins = 0
        self.total_games = 0

    def clear_history(self):
        self.reset_history()
        self.save_history()

    def get_summary(self):
        return (
            f"Ties: {self.ties}\n"
            f"Player 1 Wins: {self.player1_wins}\n"
            f"Player 2 Wins: {self.player2_wins}\n"
            f"CPU Wins: {self.cpu_wins}\n"
            f"Total Games: {self.total_games}"
        )

    def record_win(self, winner_symbol):
        if winner_symbol == 'X':
            self.player1_wins += 1
        elif winner_symbol == 'O':
            self.player2_wins += 1
        elif winner_symbol == 'CPU':
            self.cpu_wins += 1
        self.total_games += 1
        self.save_history()

    def record_tie(self):
        self.ties += 1
        self.total_games += 1
        self.save_history()
