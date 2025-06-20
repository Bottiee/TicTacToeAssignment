import os
import sqlite3
from typing import Optional

class HistoryStorage:
    """
    Manages the storage and retrieval of Tic-Tac-Toe game statistics
    in a SQLite database. This class is responsible for local data persistence.
    """

    def __init__(self, db_filename: str = 'Game_history/game_data/history.db'):
        """
        Initializes the HistoryStorage with a database filename.
        Ensures the database directory exists, connects to the database,
        creates the necessary table if it doesn't exist, and loads existing history.
        """
        self.db_filename: str = db_filename
        self._conn: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None

        # Initialize all stats to 0 before attempting to load from DB
        self.total_games: int = 0
        self.player1_wins: int = 0
        self.player2_wins: int = 0
        self.cpu_wins: int = 0
        self.ties: int = 0

        try:
            self.ensure_dir_exists()
            self.connect_db()
            self.create_table_if_not_exists()
            self.load_history()
        except Exception as e:
            print(f"ERROR: Ran into an issue during HistoryStorage setup: {e}")
            self.close() # Ensure resources are cleaned up on error

    def ensure_dir_exists(self) -> None:
        """Ensures the directory for the database file exists."""
        directory = os.path.dirname(self.db_filename)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"INFO: Created directory for DB: '{directory}'")
            except OSError as e:
                print(f"ERROR: Couldn't create directory at '{directory}': {e}. This is critical!")
                raise # Re-raise to stop initialization if directory can't be created

    def connect_db(self) -> None:
        """Establishes a connection to the SQLite database."""
        try:
            self._conn = sqlite3.connect(self.db_filename, timeout=5)
            self._cursor = self._conn.cursor()
            print(f"INFO: Database connected successfully to '{self.db_filename}'.")
        except sqlite3.Error as e:
            print(f"ERROR: Database connection failed for '{self.db_filename}': {e}")
            raise # Re-raise to be caught by the __init__ try-except

    def create_table_if_not_exists(self) -> None:
        """
        Creates the 'history' table if it does not already exist.
        The table is designed to hold a single row of game statistics (id = 1).
        """
        if not self._cursor:
            print("WARNING: Cannot create table: Database connection not established.")
            return # Or raise an error
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
            print("INFO: History table ensured.")
        except sqlite3.Error as e:
            print(f"ERROR: Problem creating the history table: {e}")
            raise

    def load_history(self) -> None:
        """Loads game statistics from the database into instance attributes."""
        if not self._cursor:
            print("WARNING: Cannot load history: Database connection not established.")
            self.reset_history() # Fallback to reset if not connected
            return

        try:
            self._cursor.execute('SELECT ties, player1_wins, player2_wins, cpu_wins, total_games FROM history WHERE id = 1 LIMIT 1')
            record = self._cursor.fetchone()
            if record is None:
                print("INFO: No history record found. Initializing new history.")
                self.reset_history()
                self.save_history()
            else:
                self.ties, self.player1_wins, self.player2_wins, self.cpu_wins, self.total_games = record
                print("INFO: History loaded successfully.")
        except sqlite3.Error as e:
            print(f"ERROR: Error trying to load game history: {e}")
            self.reset_history() # Fallback to reset on DB error
            self.save_history() # Attempt to save the reset state

    def save_history(self) -> None:
        """Saves current game statistics to the database."""
        if not self._conn:
            print("WARNING: Cannot save history: Database connection not established.")
            return
        try:
            history_data = (self.ties, self.player1_wins, self.player2_wins, self.cpu_wins, self.total_games)
            self._cursor.execute('''
                INSERT OR REPLACE INTO history (id, ties, player1_wins, player2_wins, cpu_wins, total_games)
                VALUES (1, ?, ?, ?, ?, ?)
            ''', history_data)
            self._conn.commit()
            # print("INFO: History saved successfully.") # Can be noisy, uncomment for debugging
        except sqlite3.Error as e:
            print(f"ERROR: Couldn't save history data: {e}")

    def reset_history(self) -> None:
        """Resets all game statistics to zero."""
        self.total_games = 0
        self.ties = 0
        self.player1_wins = 0
        self.player2_wins = 0
        self.cpu_wins = 0
        print("INFO: Game history reset to zero in memory.")

    def clear_history(self) -> None:
        """Clears all game history data and resets statistics."""
        print("INFO: Clearing all game history data from DB and resetting in memory...")
        self.reset_history()
        self.save_history() # Save the zeroed-out history to the DB

    def get_summary(self) -> str:
        """Returns a formatted string summary of game history."""
        return (
            f"--- Game History Summary ---\n"
            f"Games Played: {self.total_games}\n"
            f"Player One Wins: {self.player1_wins}\n"
            f"Player Two Wins: {self.player2_wins}\n"
            f"CPU Victories: {self.cpu_wins}\n"
            f"Tied Games: {self.ties}\n"
            f"----------------------------"
        )

    def record_win(self, winner_symbol: str) -> None:
        """
        Records a win for the specified winner symbol ('X', 'O', 'CPU')
        and updates total games.
        """
        if winner_symbol == 'CPU':
            self.cpu_wins += 1
        elif winner_symbol == 'X':
            self.player1_wins += 1
        elif winner_symbol == 'O':
            self.player2_wins += 1
        else:
            print(f"WARNING: Unrecognized winner symbol '{winner_symbol}'. Not recording this win.")
            return
        self.total_games += 1
        self.save_history()

    def record_tie(self) -> None:
        """Records a tie game and updates total games."""
        self.ties += 1
        self.total_games += 1
        self.save_history()

    def close(self) -> None:
        """Closes the database connection gracefully."""
        if self._conn:
            try:
                self._conn.close()
                self._conn = None
                self._cursor = None
                print("INFO: Database connection closed gracefully.")
            except sqlite3.Error as e:
                print(f"ERROR: Ran into an issue while closing the database connection: {e}")

    def __del__(self):
        """Ensures the database connection is closed when the object is garbage-collected."""
        self.close()