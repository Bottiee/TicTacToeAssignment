import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Directory where the current script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.fc2 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class TicTacToeModel(nn.Module):
    def __init__(self, board_size):
        super(TicTacToeModel, self).__init__()
        input_size = board_size * board_size
        num_filters = 512

        self.fc1 = nn.Linear(input_size, num_filters)
        self.res_blocks = nn.ModuleList([
            ResBlock(num_filters) for _ in range(3)
        ])

        self.policy_head = nn.Sequential(
            nn.Linear(num_filters, 256),
            nn.ReLU(),
            nn.Linear(256, input_size)
        )

        self.value_head = nn.Sequential(
            nn.Linear(num_filters, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        for block in self.res_blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

def board_to_tensor(board, board_size):
    mapping = {'X': 1, 'O': -1, '': 0, ' ': 0}
    flat = [mapping.get(cell, 0) for row in board for cell in row]
    tensor = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)
    return tensor

class ComputerLogic:
    def __init__(self, board_size):
        self.board_size = board_size
        self.model_filenames = {
            3: "rl_mcts_tictactoe_3x3_episode_1000.pth",
            4: "rl_mcts_tictactoe_4x4_episode_1000.pth",
            5: "rl_mcts_tictactoe_5x5_episode_3500.pth",
            6: "rl_mcts_tictactoe_6x6_episode_500.pth",
            7: "rl_mcts_tictactoe_7x7_episode_500.pth",
            8: "rl_mcts_tictactoe_8x8_episode_500.pth",
            9: "rl_mcts_tictactoe_9x9_episode_500.pth",
        }

        self.model = None
        model_full_path = None # Initialize to avoid UnboundLocalError
        if self.board_size in self.model_filenames:
            model_file_name = self.model_filenames[self.board_size]
            relative_model_path = os.path.join("..", "models_pytorch", model_file_name)
            model_full_path = os.path.normpath(os.path.join(SCRIPT_DIR, relative_model_path))
            self.model = TicTacToeModel(self.board_size)
            print(f"Attempting to load model for {self.board_size}x{self.board_size} board from: {model_full_path}")

            if os.path.exists(model_full_path):
                try:
                    state_dict = torch.load(model_full_path, map_location='cpu')
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    print(f"Successfully loaded model for {self.board_size}x{self.board_size} board.")
                except Exception as e:
                    print(f"Error loading model for {self.board_size}x{self.board_size} from {model_full_path}: {e}")
                    print("This typically means the saved model architecture does not match the current model definition, or the file is corrupted.")
                    self.model = None # Fallback to random moves if loading fails
            else:
                print(f"Model file not found for {self.board_size}x{self.board_size} board at: {model_full_path}. Using random moves.")
                self.model = None # Fallback to random moves if model_file doesn't exist
        else:
            print(f"No specific model configured for board size {self.board_size}. Using random moves.")
            self.model = None # Fallback if board_size is not in our mapping

    def get_ai_move(self, board):
        empty_cells = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if
                       board[r][c] in ['', ' ']]
        if not empty_cells:
            print("No empty cells left for AI to move.")
            return None

        if self.model:
            input_tensor = board_to_tensor(board, self.board_size)
            with torch.no_grad():
                policy_logits, _ = self.model(input_tensor)
                probs = torch.softmax(policy_logits.squeeze(0), dim=0).numpy()
            cell_indices = [r * self.board_size + c for r, c in empty_cells]

            if not cell_indices:
                print("No valid empty cell indices found for AI move.")
                move = random.choice(empty_cells)
                return move
            probs_empty = probs[cell_indices]

            if probs_empty.sum() == 0:
                 print("All probabilities for empty cells are zero. Reverting to random move.")
                 move = random.choice(empty_cells)
                 return move
            best_idx_in_probs_empty = np.argmax(probs_empty)
            best_cell_index = cell_indices[best_idx_in_probs_empty]
            move = divmod(best_cell_index, self.board_size)
            print(f"AI selected move {move} with prob {probs_empty[best_idx_in_probs_empty]:.4f}")
            return move

        else:
            move = random.choice(empty_cells)
            print(f"AI random move selected: {move}")
            return move