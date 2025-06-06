import pickle
import os
import sqlite3
from typing import Optional

class HistoryStorage:
    def __init__(self, db_filename: str = 'Game_history/game_history.db'):
        self.db_filename = db_filename
        self._conn: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None

        self.ties = 0
        self.player1_wins = 0
        self.player2_wins = 0
        self.cpu_wins = 0
        self.total_games = 0

        self.ensure_dir_exists()
        self.connect_db()
        self.create_table_if_not_exists()
        self.load_history()

    def ensure_dir_exists(self) -> None:
        directory = os.path.dirname(self.db_filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def connect_db(self) -> None:
        self._conn = sqlite3.connect(self.db_filename)
        self._cursor = self._conn.cursor()

    def create_table_if_not_exists(self) -> None:
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

    def load_history(self) -> None:
        self._cursor.execute('SELECT ties, player1_wins, player2_wins, cpu_wins, total_games FROM history WHERE id = 1')
        row = self._cursor.fetchone()
        if row is None:
            self.reset_history()
            self.save_history()
        else:
            self.ties, self.player1_wins, self.player2_wins, self.cpu_wins, self.total_games = row

    def save_history(self) -> None:
        # Try to insert or update row with id = 1
        self._cursor.execute('''
            INSERT INTO history (id, ties, player1_wins, player2_wins, cpu_wins, total_games)
            VALUES (1, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                ties=excluded.ties,
                player1_wins=excluded.player1_wins,
                player2_wins=excluded.player2_wins,
                cpu_wins=excluded.cpu_wins,
                total_games=excluded.total_games
        ''', (self.ties, self.player1_wins, self.player2_wins, self.cpu_wins, self.total_games))
        self._conn.commit()

    def reset_history(self) -> None:
        self.ties = 0
        self.player1_wins = 0
        self.player2_wins = 0
        self.cpu_wins = 0
        self.total_games = 0

    def clear_history(self) -> None:
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

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            self._cursor = None

    def __del__(self):
        self.close()
