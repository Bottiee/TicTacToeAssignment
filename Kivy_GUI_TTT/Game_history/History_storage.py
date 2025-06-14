import os
import sqlite3

class HistoryStorage:
    # A slightly less common default path, perhaps reflecting a user's choice
    def __init__(self, db_filename: str = 'game_data/history.db'):
        self.db_filename = db_filename
        # Initializing these directly instead of through Optional, then setting to None
        self._conn = None
        self._cursor = None

        # Slight reordering or grouping of attributes
        self.total_games = 0
        self.player1_wins = 0
        self.player2_wins = 0
        self.cpu_wins = 0
        self.ties = 0

        try:
            self.ensure_dir_exists()
            self.connect_db()
            self.create_table_if_not_exists()
            self.load_history()
        except Exception as e:
            print(f"Oops! Ran into an issue during HistoryStorage setup: {e}")
            self.close()

    def ensure_dir_exists(self) -> None:
        directory = os.path.dirname(self.db_filename)
        # f-string for clarity
        if directory and not os.path.exists(f"{directory}"):
            try:
                os.makedirs(directory)
            except OSError as e:
                print(f"Couldn't create directory at '{directory}': {e}. This is a problem!")
                raise

    def connect_db(self) -> None:
        try:
            self._conn = sqlite3.connect(self.db_filename, timeout=5)
            self._cursor = self._conn.cursor()
        except sqlite3.Error as e:
            print(f"Database connection failed for '{self.db_filename}': {e}")
            raise

    def create_table_if_not_exists(self) -> None:
        try:
            self._cursor.execute('''
                CREATE TABLE IF NOT EXISTS history (

                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    ties INTEGER NOT NULL,
                    player1_wins INTEGER NOT NULL,
                    player2_wins INTEGER NOT NULL,
                    cpu_wins INTEGER NOT NULL,
                    total_games INTEGER NOT NULL
                )
            ''')
            self._conn.commit()
        except sqlite3.Error as e:
            print(f"Problem creating the history table: {e}")
            raise

    def load_history(self) -> None:
        try:
            self._cursor.execute('SELECT ties, player1_wins, player2_wins, cpu_wins, total_games FROM history WHERE id = 1 LIMIT 1')
            record = self._cursor.fetchone()
            if record is None:
                self.reset_history()
                self.save_history()
            else:
                # Direct unpacking on one line
                self.ties, self.player1_wins, self.player2_wins, self.cpu_wins, self.total_games = record
        except sqlite3.Error as e:
            print(f"Error trying to load game history: {e}")
            self.reset_history() # Fallback

    def save_history(self) -> None:
        try:
            # Slightly different variable name for the tuple
            history_data = (self.ties, self.player1_wins, self.player2_wins, self.cpu_wins, self.total_games)
            self._cursor.execute('''
                INSERT OR REPLACE INTO history (id, ties, player1_wins, player2_wins, cpu_wins, total_games)
                VALUES (1, ?, ?, ?, ?, ?)
            ''', history_data) # Using the variable directly
            self._conn.commit()
        except sqlite3.Error as e:
            print(f"Couldn't save history data: {e}")

    def reset_history(self) -> None:
        # Reordered reset
        self.total_games = 0
        self.ties = 0
        self.player1_wins = 0
        self.player2_wins = 0
        self.cpu_wins = 0


    def clear_history(self) -> None:
        print("Clearing all game history data...")
        self.reset_history()
        self.save_history()

    def get_summary(self) -> str:
        return (
            f"--- Game History Summary ---\n"
            f"Games Played: {self.total_games}\n"
            f"Player One Wins: {self.player1_wins}\n"
            f"Player Two Wins: {self.player2_wins}\n"
            f"CPU Victories: {self.cpu_wins}\n" # Slightly different wording
            f"Tied Games: {self.ties}\n"
            f"----------------------------"
        )

    def record_win(self, winner_symbol: str) -> None:
        if winner_symbol == 'CPU':
            self.cpu_wins += 1
        elif winner_symbol == 'X':
            self.player1_wins += 1
        elif winner_symbol == 'O':
            self.player2_wins += 1
        else:
            print(f"Warning: Unrecognized winner symbol '{winner_symbol}'. Not recording this win.")
            return
        self.total_games += 1
        self.save_history()

    def record_tie(self) -> None:
        self.ties += 1
        self.total_games += 1
        self.save_history()

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
                self._conn = None
                self._cursor = None
                print("Database connection closed gracefully.")
            except sqlite3.Error as e:
                print(f"Ran into an issue while closing the database connection: {e}")

    def __del__(self):
        self.close()