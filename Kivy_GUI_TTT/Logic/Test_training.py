import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os
import math

def check_win_condition(board, symbol, consecutive_needed):
    size = len(board)

    def check_line(line, target_symbol, count_needed):
        current_consecutive = 0
        for cell in line:
            if cell == target_symbol:
                current_consecutive += 1
                if current_consecutive == count_needed:
                    return True
            else:
                current_consecutive = 0
        return False

    for r in range(size): # Check rows
        if check_line(board[r], symbol, consecutive_needed):
            return True

    for c in range(size): # Check columns
        column = [board[r][c] for r in range(size)]
        if check_line(column, symbol, consecutive_needed):
            return True

    for k in range(-(size - consecutive_needed), size - (consecutive_needed - 1)): # Check main diagonals
        diagonal = [board[i][i - k] for i in range(max(0, k), min(size, size + k))]
        if check_line(diagonal, symbol, consecutive_needed):
            return True

    for k in range(consecutive_needed - 1, 2 * size - consecutive_needed): # Check anti-diagonals
        diagonal = [board[i][k - i] for i in range(max(0, k - size + 1), min(size, k + 1))]
        if check_line(diagonal, symbol, consecutive_needed):
            return True
    return False

def check_draw_condition(board):
    return all(cell != '' for row in board for cell in row)

def get_available_moves(board):
    available = []
    size = len(board)
    for r in range(size):
        for c in range(size):
            if board[r][c] == '':
                available.append((r, c))
    return available



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

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, board_size):
        super(QNetwork, self).__init__()
        self.board_size = board_size
        self.fc1 = nn.Linear(input_size, 512)
        self.res_blocks = nn.ModuleList([
            ResBlock(512) for _ in range(3)
        ])
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        for block in self.res_blocks:
            x = block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class MCTSNode:
    def __init__(self, board, player_symbol, parent=None, parent_action=None):
        self.board = board
        self.player_symbol = player_symbol
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}
        self.visit_counts = {}
        self.action_values = {}
        self.prior_probabilities = {}
        self.total_visits = 0
        self.is_expanded = False

    def select_child(self, c_puct):
        best_action = None
        max_uct_value = -float('inf')
        available_moves = get_available_moves(self.board)

        for action in available_moves:
            if action not in self.visit_counts or self.visit_counts[action] == 0:
                uct_value = c_puct * self.prior_probabilities.get(action, 0.0) * math.sqrt(self.total_visits + 1e-8) / (1 + 1e-8)
            else:
                Q_sa = self.action_values[action]
                N_sa = self.visit_counts[action]
                P_sa = self.prior_probabilities.get(action, 0.0)
                uct_value = Q_sa + c_puct * P_sa * math.sqrt(self.total_visits) / (1 + N_sa)

            if uct_value > max_uct_value:
                max_uct_value = uct_value
                best_action = action
        return best_action

    def expand(self, policy_logits, value_estimate, available_moves, board_size):
        self.is_expanded = True
        self.total_visits = 1
        action_to_flat_idx = {(r, c): r * board_size + c for r in range(board_size) for c in range(board_size)}

        logits_for_available_moves = [policy_logits[action_to_flat_idx[move]].item() for move in available_moves]

        exp_logits = np.exp(logits_for_available_moves - np.max(logits_for_available_moves))
        prior_probs_normalized = exp_logits / np.sum(exp_logits + 1e-8)

        for i, action in enumerate(available_moves):
            self.visit_counts[action] = 0
            self.action_values[action] = 0.0
            self.prior_probabilities[action] = prior_probs_normalized[i]
        return value_estimate

    def backpropagate(self, value):
        node = self
        current_value_to_propagate = value

        while node is not None:
            node.total_visits += 1
            if node.parent is not None and node.parent_action is not None:
                parent_node = node.parent
                parent_action = node.parent_action

                if parent_action in parent_node.visit_counts:
                    parent_node.visit_counts[parent_action] += 1
                    N_sa = parent_node.visit_counts[parent_action]
                    Q_sa_old = parent_node.action_values[parent_action]
                    parent_node.action_values[parent_action] = ((Q_sa_old * (N_sa - 1)) + (-current_value_to_propagate)) / N_sa
                else:
                    print(f"Warning: Action {parent_action} not found in parent's visit_counts during backprop.")
            node = node.parent
            current_value_to_propagate *= -1


class RLPlayer:
    def __init__(self, player_symbol, opponent_symbol, board_size, c_puct=1.0, num_simulations=100):
        self.player_symbol = player_symbol
        self.opponent_symbol = opponent_symbol
        self.board_size = board_size
        self.input_size = board_size * board_size
        self.output_size = board_size * board_size
        self.c_puct = c_puct
        self.num_simulations = num_simulations

        self.device = torch.device("cpu")
        device_found_message = "No compatible GPU found, falling back to CPU. Training will be slower."
        try:
            import torch_directml
            if torch_directml.is_available():
                self.device = torch_directml.device()
                device_found_message = "Using DirectML GPU (for Windows AMD/Intel)."
        except ImportError:
            pass
        if self.device.type == "cpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_found_message = f"Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}"
        elif self.device.type == "cpu" and hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
            self.device = torch.device("hip")
            device_found_message = f"Using AMD ROCm GPU: {torch.hip.get_device_name(0)}"
        elif self.device.type == "cpu" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_found_message = "Using Apple Silicon MPS GPU."
        print(device_found_message)

        self.policy_model = QNetwork(self.input_size, self.output_size, self.board_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=0.001)
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=200000)
        self.gamma = 0.99
        self.batch_size = 128

    def _state_to_input(self, board):
        encoded_board = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        for r_idx in range(self.board_size):
            for c_idx in range(self.board_size):
                flat_idx = r_idx * self.board_size + c_idx
                if board[r_idx][c_idx] == self.player_symbol:
                    encoded_board[flat_idx] = 1.0
                elif board[r_idx][c_idx] == self.opponent_symbol:
                    encoded_board[flat_idx] = -1.0
        return torch.from_numpy(encoded_board).float().unsqueeze(0).to(self.device)

    def get_move(self, board, available_moves, current_player_symbol):
        if not available_moves:
            return None

        self.policy_model.eval()
        root = MCTSNode(board, current_player_symbol)

        with torch.no_grad():
            policy_logits_raw, value_raw = self.policy_model(self._state_to_input(root.board))
        root.expand(policy_logits_raw.squeeze(0).cpu().numpy(), value_raw.item(), available_moves, self.board_size)

        for sim in range(self.num_simulations):
            node = root
            path = [root]
            while node.is_expanded and get_available_moves(node.board):
                action = node.select_child(self.c_puct)
                if action is None:
                    break
                next_board = [row[:] for row in node.board]
                next_player_symbol = self.opponent_symbol if node.player_symbol == self.player_symbol else self.player_symbol
                next_board[action[0]][action[1]] = node.player_symbol

                is_win = check_win_condition(next_board, node.player_symbol, consecutive_needed=CONSECUTIVE_NEEDED)
                is_draw = check_draw_condition(next_board) and not is_win

                if is_win:
                    value = 1.0
                    node.backpropagate(value)
                    break
                elif is_draw:
                    value = 0.0
                    node.backpropagate(value)
                    break

                if action not in node.children:
                    node.children[action] = MCTSNode(next_board, next_player_symbol, node, action)
                node = node.children[action]
                path.append(node)
            else:
                if not get_available_moves(node.board):
                    if check_win_condition(node.board, self.player_symbol, consecutive_needed=CONSECUTIVE_NEEDED):
                        value = 1.0
                    elif check_win_condition(node.board, self.opponent_symbol, consecutive_needed=CONSECUTIVE_NEEDED):
                        value = -1.0
                    else:
                        value = 0.0
                else:
                    with torch.no_grad():
                        policy_logits_raw, value_raw = self.policy_model(self._state_to_input(node.board))
                    value = node.expand(policy_logits_raw.squeeze(0).cpu().numpy(), value_raw.item(),
                                        get_available_moves(node.board), self.board_size)
                node.backpropagate(value)

        final_available_moves = get_available_moves(root.board)
        mcts_policy = np.zeros(self.output_size)
        total_root_visits = 0

        for action in final_available_moves:
            if action in root.visit_counts:
                idx = action[0] * self.board_size + action[1]
                mcts_policy[idx] = root.visit_counts[action]
                total_root_visits += root.visit_counts[action]

        if total_root_visits > 0:
            mcts_policy /= total_root_visits
        else:
            if final_available_moves:
                print("Warning: MCTS total root visits is 0, falling back to uniform policy.")
                for move in final_available_moves:
                    idx = move[0] * self.board_size + move[1]
                    mcts_policy[idx] = 1.0 / len(final_available_moves)
            else:
                return None, None

        best_action = None
        max_visits = -1
        for action in final_available_moves:
            if action in root.visit_counts and root.visit_counts[action] > max_visits:
                max_visits = root.visit_counts[action]
                best_action = action

        if best_action is None and final_available_moves:
            print("Warning: MCTS couldn't find a preferred move after simulations, picking random.")
            best_action = random.choice(final_available_moves)
        self.policy_model.train()
        return best_action, mcts_policy

    def remember(self, state, mcts_policy_target, game_result_target):
        self.replay_buffer.append(([row[:] for row in state], mcts_policy_target, game_result_target))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.policy_model.train()
        batch = random.sample(self.replay_buffer, self.batch_size)
        states_batch = torch.cat([self._state_to_input(s) for s in [exp[0] for exp in batch]]).to(self.device)
        policy_targets_batch = torch.tensor(np.array([exp[1] for exp in batch]), dtype=torch.float32).to(self.device)
        value_targets_batch = torch.tensor(np.array([exp[2] for exp in batch]), dtype=torch.float32).unsqueeze(1).to(self.device)

        predicted_policy_logits, predicted_value = self.policy_model(states_batch)
        policy_loss = self.policy_loss_fn(predicted_policy_logits, policy_targets_batch)
        value_loss = self.value_loss_fn(predicted_value, value_targets_batch)
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


def train_agent(agent, num_episodes, consecutive_needed):
    for episode in range(num_episodes):
        board = [['' for _ in range(agent.board_size)] for _ in range(agent.board_size)]
        done = False
        game_history = []
        current_player_symbol = 'X'

        while not done:
            available_moves = get_available_moves(board)
            if not available_moves:
                game_result = 0
                done = True
                break

            chosen_action, mcts_policy = agent.get_move(board, available_moves, current_player_symbol)
            if chosen_action is None or mcts_policy is None:
                print(f"Episode {episode}: MCTS failed to pick a move despite available moves or policy. Forcing Draw.")
                game_result = 0
                done = True
                break

            game_history.append((
                [row[:] for row in board],
                mcts_policy,
                current_player_symbol
            ))

            board[chosen_action[0]][chosen_action[1]] = current_player_symbol

            if check_win_condition(board, current_player_symbol, consecutive_needed=consecutive_needed):
                game_result_val = 1
                done = True
            elif check_draw_condition(board):
                game_result_val = 0
                done = True
            else:
                current_player_symbol = 'O' if current_player_symbol == 'X' else 'X'
                game_result_val = None

            if done:
                for state, mcts_pol, player_at_turn in game_history:
                    target_value = game_result_val
                    if player_at_turn == agent.opponent_symbol:
                        target_value *= -1
                    agent.remember(state, mcts_pol, target_value)

                outcome_str = "Draw"
                if game_result_val == 1:
                    outcome_str = f"{game_history[-1][2]} Wins!"
                elif game_result_val == -1:
                    outcome_str = f"{agent.opponent_symbol if game_history[-1][2] == agent.player_symbol else agent.player_symbol} Wins!"

                print(
                    f"Episode {episode}: {outcome_str}. Replay Buffer Size: {len(agent.replay_buffer)}. Total MCTS Simulations: {agent.num_simulations}")
                break
            agent.learn()

        if episode % 500 == 0 and episode > 0:
            if not os.path.exists("../models_pytorch"):
                os.makedirs("../models_pytorch")
            model_path = f"../models_pytorch/rl_mcts_tictactoe_{agent.board_size}x{agent.board_size}_episode_{episode}.pth"
            torch.save(agent.policy_model.state_dict(), model_path)
            print(f"Model weights saved to {model_path}")


if __name__ == "__main__":
    BOARD_DIM = 5
    CONSECUTIVE_NEEDED = BOARD_DIM
    NUM_EPISODES = 1000
    C_PUCT_CONST = 1.0
    MCTS_SIMULATIONS = 20

    player_agent = RLPlayer('X', 'O', BOARD_DIM, c_puct=C_PUCT_CONST, num_simulations=MCTS_SIMULATIONS)

    print(f"\n--- Starting Training for {BOARD_DIM}x{BOARD_DIM} Board with MCTS ---")
    print(f"Winning condition: {CONSECUTIVE_NEEDED}-in-a-row (same as board dimension)")
    print(f"Using device: {player_agent.device}")
    print(f"Replay Buffer Size: {player_agent.replay_buffer.maxlen}")
    print(f"Batch Size: {player_agent.batch_size}")
    print(f"MCTS Simulations per Move: {player_agent.num_simulations}")
    print(f"MCTS C_PUCT Constant: {player_agent.c_puct}")
    print(
        "This training will be significantly more computationally intensive per episode than the 3x3 DQN, especially with high CONSECUTIVE_NEEDED.")

    train_agent(player_agent, num_episodes=NUM_EPISODES, consecutive_needed=CONSECUTIVE_NEEDED)

    print("\n--- Training Complete ---")