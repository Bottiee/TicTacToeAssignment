import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os
import math


# --- Game Logic Functions (from your provided code) ---
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

    # Check rows
    for r in range(size):
        if check_line(board[r], symbol, consecutive_needed):
            return True

    # Check columns
    for c in range(size):
        column = [board[r][c] for r in range(size)]
        if check_line(column, symbol, consecutive_needed):
            return True

    # Check main diagonals (top-left to bottom-right)
    for k in range(-(size - consecutive_needed), size - (consecutive_needed - 1)):
        diagonal = []
        for i in range(size):
            if 0 <= i - k < size:
                diagonal.append(board[i][i - k])
        if check_line(diagonal, symbol, consecutive_needed):
            return True

    # Check anti-diagonals (top-right to bottom-left)
    for k in range(consecutive_needed - 1, 2 * size - consecutive_needed):
        diagonal = []
        for i in range(size):
            if 0 <= k - i < size:
                diagonal.append(board[i][k - i])
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


# --- Neural Network Architecture (from your provided code) ---
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


# --- MCTS Node Implementation (from your provided code) ---
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
            # Add a small epsilon to the denominator to prevent division by zero, especially for unvisited nodes
            denominator = 1 + self.visit_counts.get(action, 0)
            numerator_sqrt = math.sqrt(self.total_visits)

            if action not in self.visit_counts or self.visit_counts[action] == 0:
                # For unvisited nodes, Q_sa is 0, so UCT simplifies
                uct_value = c_puct * self.prior_probabilities.get(action, 0.0) * numerator_sqrt / denominator
            else:
                Q_sa = self.action_values[action]
                N_sa = self.visit_counts[action]
                P_sa = self.prior_probabilities.get(action, 0.0)
                uct_value = Q_sa + c_puct * P_sa * numerator_sqrt / denominator

            if uct_value > max_uct_value:
                max_uct_value = uct_value
                best_action = action
        return best_action

    def expand(self, policy_logits, value_estimate, available_moves, board_size):
        self.is_expanded = True
        self.total_visits = 1  # Node is visited once it's expanded
        action_to_flat_idx = {(r, c): r * board_size + c for r in range(board_size) for c in range(board_size)}

        logits_for_available_moves = [policy_logits[action_to_flat_idx[move]].item() for move in available_moves]

        # Apply softmax to convert logits to probabilities
        exp_logits = np.exp(
            logits_for_available_moves - np.max(logits_for_available_moves))  # Subtract max for numerical stability
        prior_probs_normalized = exp_logits / np.sum(exp_logits + 1e-8)  # Add small epsilon to denominator

        for i, action in enumerate(available_moves):
            self.visit_counts[action] = 0
            self.action_values[action] = 0.0
            self.prior_probabilities[action] = prior_probs_normalized[i]
        return value_estimate  # This value is from the perspective of the player whose turn it is at this node

    def backpropagate(self, value):
        node = self
        current_value_to_propagate = value

        while node is not None:
            node.total_visits += 1
            if node.parent is not None and node.parent_action is not None:
                parent_node = node.parent
                parent_action = node.parent_action

                if parent_action in parent_node.visit_counts:
                    # Update Q-value: Q(s,a) = (Q(s,a)*(N(s,a)-1) + V) / N(s,a)
                    # The value for the parent is the negative of the child's value
                    parent_node.action_values[parent_action] = \
                        ((parent_node.action_values[parent_action] * (parent_node.visit_counts[parent_action])) + (
                            -current_value_to_propagate)) / \
                        (parent_node.visit_counts[parent_action] + 1)
                    parent_node.visit_counts[parent_action] += 1
                # else: This case implies an error in tree traversal logic if action isn't in visit_counts
            node = node.parent
            current_value_to_propagate *= -1  # Flip value for the next parent's perspective


# --- RLPlayer and Training Logic (from your provided code, slightly refined) ---
class RLPlayer:
    def __init__(self, player_symbol, opponent_symbol, board_size, c_puct=1.0, num_simulations=100):
        self.player_symbol = player_symbol
        self.opponent_symbol = opponent_symbol
        self.board_size = board_size
        self.input_size = board_size * board_size
        self.output_size = board_size * board_size
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.consecutive_needed = 3  # Fixed to 3 for classical Tic-Tac-Toe win condition

        self.device = torch.device("cpu")
        device_found_message = "No compatible GPU found, falling back to CPU. Training will be slower."
        try:
            import torch_directml
            if torch_directml.is_available():
                self.device = torch_directml.device()
                device_found_message = "Using DirectML GPU (for Windows AMD/Intel)."
        except ImportError:
            pass  # No directml installed, continue
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
        self.gamma = 0.99  # Gamma is less relevant for MCTS, but kept for consistency if needed later
        self.batch_size = 128

    def _state_to_input(self, board):
        """
        Converts the board (list of lists) into a numerical tensor input for the neural network.
        Player's symbol: 1.0, Opponent's symbol: -1.0, Empty: 0.0
        Reshapes to (1, board_size*board_size) for a flattened input layer.
        """
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
            return None, None

        self.policy_model.eval()  # Set model to evaluation mode
        root = MCTSNode(board, current_player_symbol)

        with torch.no_grad():
            policy_logits_raw, value_raw = self.policy_model(self._state_to_input(root.board))

        # Expand the root node based on initial policy and value prediction
        root.expand(policy_logits_raw.squeeze(0).cpu().numpy(), value_raw.item(), available_moves, self.board_size)

        for sim in range(self.num_simulations):
            node = root
            path = [root]  # Track path for backpropagation

            # --- Selection Phase ---
            while node.is_expanded and get_available_moves(node.board):
                action = node.select_child(self.c_puct)
                if action is None:  # No valid children to select (e.g., all explored or no moves left)
                    break

                # Simulate applying the action
                next_board_state = [row[:] for row in node.board]
                player_making_sim_move = node.player_symbol  # The player whose turn it is AT THIS NODE
                next_board_state[action[0]][action[1]] = player_making_sim_move

                # Determine next player for the child node
                next_player_symbol_in_tree = self.opponent_symbol if player_making_sim_move == self.player_symbol else self.player_symbol

                # Check for immediate win/draw after this simulated move
                is_win_after_simulated_move = check_win_condition(next_board_state, player_making_sim_move,
                                                                  self.consecutive_needed)
                is_draw_after_simulated_move = check_draw_condition(
                    next_board_state) and not is_win_after_simulated_move

                if is_win_after_simulated_move:
                    value = 1.0  # Win for the player who just made the move
                    node.backpropagate(value)
                    break  # End simulation, backpropagate
                elif is_draw_after_simulated_move:
                    value = 0.0  # Draw
                    node.backpropagate(value)
                    break  # End simulation, backpropagate

                # If game not over, move to the child node
                if action not in node.children:
                    node.children[action] = MCTSNode(next_board_state, next_player_symbol_in_tree, node, action)
                node = node.children[action]
                path.append(node)
            else:  # If loop finished because node not expanded or no more moves
                # --- Expansion and Evaluation Phase ---
                value = 0.0  # Default value for terminal draw

                # Check if the node's board is a terminal state
                if not get_available_moves(node.board):  # Board full
                    if check_win_condition(node.board, self.player_symbol, self.consecutive_needed):
                        value = 1.0 if node.player_symbol == self.player_symbol else -1.0  # Value from perspective of player_symbol at node
                    elif check_win_condition(node.board, self.opponent_symbol, self.consecutive_needed):
                        value = 1.0 if node.player_symbol == self.opponent_symbol else -1.0
                    else:  # Full board, no winner
                        value = 0.0
                elif check_win_condition(node.board, node.player_symbol,
                                         self.consecutive_needed):  # Current node's player wins
                    value = 1.0
                elif check_win_condition(node.board, self.opponent_symbol,
                                         self.consecutive_needed):  # Opponent of current node's player wins
                    value = -1.0
                else:  # Non-terminal node, evaluate with NN
                    with torch.no_grad():
                        policy_logits_raw, value_raw = self.policy_model(self._state_to_input(node.board))
                    value = node.expand(policy_logits_raw.squeeze(0).cpu().numpy(), value_raw.item(),
                                        get_available_moves(node.board), self.board_size)

                # --- Backpropagation Phase ---
                node.backpropagate(value)

        # --- Select Best Move from Root's Visit Counts ---
        final_available_moves = get_available_moves(root.board)
        mcts_policy = np.zeros(self.output_size)
        total_root_visits = 0

        # Populate MCTS policy based on visit counts
        for action in final_available_moves:
            if action in root.visit_counts:
                idx = action[0] * self.board_size + action[1]
                mcts_policy[idx] = root.visit_counts[action]
                total_root_visits += root.visit_counts[action]

        # Normalize MCTS policy
        if total_root_visits > 0:
            mcts_policy /= total_root_visits
        else:  # Fallback if MCTS somehow yielded no visits
            if final_available_moves:
                for move in final_available_moves:
                    idx = move[0] * self.board_size + move[1]
                    mcts_policy[idx] = 1.0 / len(final_available_moves)
            else:
                return None, None  # No moves, no policy

        # Choose the action with the most visits
        best_action = None
        max_visits = -1
        for action in final_available_moves:
            if action in root.visit_counts and root.visit_counts[action] > max_visits:
                max_visits = root.visit_counts[action]
                best_action = action

        if best_action is None and final_available_moves:
            # Fallback for extreme edge cases where no visits accumulated for valid moves
            best_action = random.choice(final_available_moves)
        elif best_action is None:  # No moves at all
            return None, None

        self.policy_model.train()  # Set model back to training mode
        return best_action, mcts_policy

    def remember(self, state, mcts_policy_target, game_result_target):
        """Stores a game experience for replay."""
        # Store raw board state (list of lists) along with MCTS policy and game result
        self.replay_buffer.append(([row[:] for row in state], mcts_policy_target, game_result_target))

    def learn(self):
        """Trains the neural network using a batch from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return

        self.policy_model.train()  # Set model to training mode
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Prepare batch data for PyTorch
        states_batch = torch.cat([self._state_to_input(s) for s in [exp[0] for exp in batch]]).to(self.device)
        policy_targets_batch = torch.tensor(np.array([exp[1] for exp in batch]), dtype=torch.float32).to(self.device)
        value_targets_batch = torch.tensor(np.array([exp[2] for exp in batch]), dtype=torch.float32).unsqueeze(1).to(
            self.device)

        # Forward pass
        predicted_policy_logits, predicted_value = self.policy_model(states_batch)

        # Calculate loss
        policy_loss = self.policy_loss_fn(predicted_policy_logits, policy_targets_batch)
        value_loss = self.value_loss_fn(predicted_value, value_targets_batch)
        total_loss = policy_loss + value_loss

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


# --- Training Script (from your provided code, slightly refined) ---
def train_agent(agent, num_episodes):
    """
    Trains the RL agent through multiple episodes of the game using MCTS.

    Args:
        agent (RLPlayer): The reinforcement learning agent.
        num_episodes (int): The total number of episodes to train for.
    """
    # No target model updates needed explicitly here as MCTS doesn't use a separate target network in this alphaGo-like setup.
    # The MCTS itself provides the 'target' policy and value.

    for episode in range(num_episodes):
        board = [['' for _ in range(agent.board_size)] for _ in range(agent.board_size)]
        done = False
        game_history = []  # Stores (board_state, mcts_policy, player_at_turn) for each step
        current_player_symbol = 'X'  # Start with 'X' as the first player

        while not done:
            available_moves = get_available_moves(board)

            # Check for immediate draw before making a move
            if not available_moves:
                game_result_val = 0  # Draw
                done = True
                break

            # Agent makes a move using MCTS
            # MCTS always simulates from the perspective of 'current_player_symbol'
            chosen_action, mcts_policy = agent.get_move(board, available_moves, current_player_symbol)

            if chosen_action is None:  # Should only happen if no moves available
                game_result_val = 0  # Draw
                done = True
                break

            # Store the current state, the policy generated by MCTS, and the current player
            game_history.append((
                [row[:] for row in board],  # Board state *before* the action was applied
                mcts_policy,  # The policy derived from MCTS for this state
                current_player_symbol  # The player who was about to move (whose turn it was)
            ))

            # Apply the chosen action to the board
            board[chosen_action[0]][chosen_action[1]] = current_player_symbol

            # Check game outcome after the move
            if check_win_condition(board, current_player_symbol, consecutive_needed=agent.consecutive_needed):
                game_result_val = 1  # Current player wins
                done = True
            elif check_draw_condition(board):
                game_result_val = 0  # Draw
                done = True
            else:
                # Game continues, switch player for the next turn
                current_player_symbol = 'O' if current_player_symbol == 'X' else 'X'
                game_result_val = None  # Game not finished yet

            if done:
                # Once the game is over, iterate through the game history and store transitions
                # with the final outcome (reward) from the perspective of each player
                for state_hist, mcts_pol_hist, player_at_turn_hist in game_history:
                    # `game_result_val` is from the perspective of the *winning player* (1, -1, 0)
                    # We need to adjust it for the perspective of `player_at_turn_hist`
                    target_value_for_agent = game_result_val  # If current_player_symbol won
                    if player_at_turn_hist != current_player_symbol:  # If it was opponent's turn in history
                        target_value_for_agent *= -1  # Flip value for opponent's perspective

                    agent.remember(state_hist, mcts_pol_hist, target_value_for_agent)
                break  # Exit the `while not done` loop

            agent.learn()  # The agent learns after each step within an episode

        # Log episode results and save model weights periodically
        if (episode + 1) % 500 == 0:
            if not os.path.exists("../models_pytorch"):
                os.makedirs("../models_pytorch")
            model_path = f"../models_pytorch/rl_mcts_tictactoe_{agent.board_size}x{agent.board_size}_episode_{episode + 1}.pth"
            torch.save(agent.policy_model.state_dict(), model_path)
            print(f"Model weights saved to {model_path}")

        outcome_str = ""
        if game_result_val == 1:
            outcome_str = f"{current_player_symbol} Wins!"  # `current_player_symbol` is the one who made the winning move
        elif game_result_val == -1:
            winning_player_sym = 'O' if current_player_symbol == 'X' else 'X'  # The one who didn't just move, but won
            outcome_str = f"{winning_player_sym} Wins!"
        else:
            outcome_str = "Draw"

        print(
            f"Episode {episode + 1}/{num_episodes}: {outcome_str}. Replay Buffer Size: {len(agent.replay_buffer)}. MCTS Simulations: {agent.num_simulations}")


# --- Main Execution Block ---
if __name__ == "__main__":
    BOARD_DIM = 5
    CONSECUTIVE_NEEDED_FOR_WIN = 3  # Fixed for classical Tic-Tac-Toe
    NUM_EPISODES = 500
    C_PUCT_CONST = 1.0
    MCTS_SIMULATIONS = 50

    player_agent = RLPlayer('X', 'O', BOARD_DIM, c_puct=C_PUCT_CONST, num_simulations=MCTS_SIMULATIONS)

    print(f"\n--- Starting Training for {BOARD_DIM}x{BOARD_DIM} Board with MCTS ---")
    print(f"Winning condition: {player_agent.consecutive_needed}-in-a-row (classical Tic-Tac-Toe)")
    print(f"Using device: {player_agent.device}")
    print(f"Replay Buffer Size: {player_agent.replay_buffer.maxlen}")
    print(f"Batch Size: {player_agent.batch_size}")
    print(f"MCTS Simulations per Move: {player_agent.num_simulations}")
    print(f"MCTS C_PUCT Constant: {player_agent.c_puct}")
    print("This MCTS training is computationally intensive, especially for larger board sizes.")

    train_agent(player_agent, num_episodes=NUM_EPISODES)

    print("\n--- Training Complete ---")