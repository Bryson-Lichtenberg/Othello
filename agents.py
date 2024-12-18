# agents.py
from evaluation import evaluate_board

from math import sqrt, log
import random

class MCTSNode:
    def __init__(self, board, parent=None, move=None, c=1.45):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0
        self.children = []
        self.c = c

    def ucb1(self, exploration_param=1.45):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration_param * sqrt(log(self.parent.visits) / self.visits)

def monte_carlo(board, color, simulations=1000, exploration_param=1.45):
    root = MCTSNode(board)

    # print(f"Starting Monte Carlo Tree Search with {int(simulations)} simulations...")
    for sim in range(int(simulations)):  # Ensure simulations is an integer
        # if sim % 10000 == 0:  # Print progress every 100 simulations
        #     print(f"Simulation {sim}/{simulations}")

        # Selection
        node = root
        while node.children:
            node = max(node.children, key=lambda n: n.ucb1(exploration_param))
        # print(f"Selected node with move {node.move} and UCB1 value: {node.ucb1(exploration_param)}")

        # Expansion
        if not board.is_game_over():
            valid_moves = board.get_valid_moves(node.board.current_turn)
            # if valid_moves:
                # print(f"Expanding node with {len(valid_moves)} valid moves...")
            for move in valid_moves:
                new_board = node.board.copy()
                new_board.execute_move(move[0], move[1], node.board.current_turn)
                child_node = MCTSNode(new_board, parent=node, move=move)
                node.children.append(child_node)

        # Simulation
        if node.children:
            node = random.choice(node.children)
        result = simulate_rollout(node.board, color)
        # print(f"Simulated rollout result: {'win' if result == 1 else 'loss' if result == 0 else 'draw'}")

        # Backpropagation
        while node:
            node.visits += 1
            if node.board.current_turn == color:
                node.wins += result
            else:
                node.wins += (1 - result)
            # print(f"Backpropagating at node with move {node.move}: visits={node.visits}, wins={node.wins}")
            node = node.parent

    # Return the move with the highest visits
    best_move = max(root.children, key=lambda n: n.visits).move
    # print(f"Best move determined: {best_move}")
    return best_move



def simulate_rollout(board, color):
    current_turn = board.current_turn
    while not board.is_game_over():
        valid_moves = board.get_valid_moves(current_turn)
        if not valid_moves:
            current_turn = -current_turn
            continue
        move = random.choice(valid_moves)
        board.execute_move(move[0], move[1], current_turn)
        current_turn = -current_turn

    winner, _ = board.get_winner()
    return 1 if winner == color else 0


def naive(board, color, heuristic):
    # Find the best move using the given heuristic at depth 1.
    best_score = float('-inf')
    best_move = None

    for move in board.get_valid_moves(color):
        row, col = move
        # Make a hypothetical move.
        board_copy = [row[:] for row in board.board]
        board.execute_move(row, col, color)
        score = evaluate_board(board, heuristic)
        board.board = board_copy  # Undo the move.

        if score > best_score:
            best_score = score
            best_move = move

    return best_move if best_move else (-1, -1)


def minimax(board, depth, maximizing_player, heuristic='naive'):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board, heuristic), None
    
    valid_moves = board.get_valid_moves(board.current_turn)
    if not valid_moves:
        return evaluate_board(board, heuristic), None

    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        for move in valid_moves:
            row, col = move
            # Make a hypothetical move
            board_copy = [row[:] for row in board.board]
            board.execute_move(row, col, board.current_turn)
            eval, _ = minimax(board, depth - 1, False, heuristic)
            board.board = board_copy  # Undo move
            if eval > max_eval:
                max_eval = eval
                best_move = move
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in valid_moves:
            row, col = move
            # Make a hypothetical move
            board_copy = [row[:] for row in board.board]
            board.execute_move(row, col, board.current_turn)
            eval, _ = minimax(board, depth - 1, True, heuristic)
            board.board = board_copy  # Undo move
            if eval < min_eval:
                min_eval = eval
                best_move = move
        return min_eval, best_move

def alpha_beta_minimax(board, depth, alpha, beta, maximizing_player, heuristic='naive'):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board, heuristic), None

    valid_moves = board.get_valid_moves(board.current_turn)
    if not valid_moves:
        return evaluate_board(board, heuristic), None

    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        for move in valid_moves:
            row, col = move
            # Make a hypothetical move
            board_copy = [row[:] for row in board.board]
            board.execute_move(row, col, board.current_turn)
            eval, _ = alpha_beta_minimax(board, depth - 1, alpha, beta, False, heuristic)
            board.board = board_copy  # Undo move
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in valid_moves:
            row, col = move
            # Make a hypothetical move
            board_copy = [row[:] for row in board.board]
            board.execute_move(row, col, board.current_turn)
            eval, _ = alpha_beta_minimax(board, depth - 1, alpha, beta, True, heuristic)
            board.board = board_copy  # Undo move
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move
    

import time

def iterative_deepening_minimax(board, max_time, maximizing_player, heuristic='naive'):
    start_time = time.time()
    depth = 1
    best_move = None

    while True:
        if time.time() - start_time > max_time:
            break

        eval, move = minimax(board, depth, maximizing_player, heuristic)
        if move:
            best_move = move

        depth += 1

    return best_move

# Similarly, for alpha-beta minimax:
def iterative_deepening_alpha_beta_minimax(board, max_time, maximizing_player, heuristic='naive'):
    start_time = time.time()
    depth = 1
    best_move = None

    while True:
        if time.time() - start_time > max_time:
            break

        eval, move = alpha_beta_minimax(board, depth, float('-inf'), float('inf'), maximizing_player, heuristic)
        if move:
            best_move = move

        depth += 1

    return best_move


def expectiminimax(board, depth, color, heuristic='naive'):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board, heuristic), None

    if board.current_turn == color:  # Maximize for the agent
        max_eval = float('-inf')
        best_move = None
        for move in board.get_valid_moves(color):
            new_board = board.copy()
            new_board.execute_move(move[0], move[1], color)
            eval, _ = expectiminimax(new_board, depth - 1, color, heuristic)
            if eval > max_eval:
                max_eval = eval
                best_move = move

        # If no valid move found, use alpha-beta pruning as a backup
        if best_move is None:
            _, best_move = alpha_beta_minimax(board, depth, float('-inf'), float('inf'), True, heuristic)

        return max_eval, best_move

    elif board.current_turn == -color:  # Minimize for the opponent
        min_eval = float('inf')
        best_move = None
        for move in board.get_valid_moves(-color):
            new_board = board.copy()
            new_board.execute_move(move[0], move[1], -color)
            eval, _ = expectiminimax(new_board, depth - 1, color, heuristic)
            if eval < min_eval:
                min_eval = eval
                best_move = move

        # If no valid move found, use alpha-beta pruning as a backup
        if best_move is None:
            _, best_move = alpha_beta_minimax(board, depth, float('-inf'), float('inf'), False, heuristic)

        return min_eval, best_move

    else:  # Chance node
        expected_value = 0
        best_move = None
        moves = board.get_valid_moves(board.current_turn)
        total_capture_value = sum(board.get_move_score(move[0], move[1], board.current_turn) for move in moves)

        for move in moves:
            move_probability = board.get_move_score(move[0], move[1], board.current_turn) / total_capture_value
            new_board = board.copy()
            new_board.execute_move(move[0], move[1], board.current_turn)
            eval, _ = expectiminimax(new_board, depth - 1, color, heuristic)
            expected_value += move_probability * eval

        # If no valid move found, use alpha-beta pruning as a backup
        if best_move is None:
            _, best_move = alpha_beta_minimax(board, depth, float('-inf'), float('inf'), board.current_turn == color, heuristic)

        return expected_value, best_move




import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import load_model



def mlp(board, model):
    """
    Use a trained MLP to determine the best move on the board.
    
    Args:
        board: An instance of the game board with `vectorize_board` and `get_mask` methods.
        mlp_path (str): Path to the trained MLP model.
    
    Returns:
        tuple: (row, col) coordinates of the best move on the 8x8 board.
    """
    if not any(board.get_mask(board.current_turn)):
        print(f"No valid moves for {'Black' if board.current_turn == 1 else 'White'}, skipping turn.")
        return -1, -1  # Signal to skip turn


    # Get the input vector by vectorizing the board
    input_vector = board.vectorize_board()
    input_vector_reshaped = input_vector.reshape(1, -1)  # Reshape to (1, 65) for the MLP
    assert input_vector_reshaped.shape == (1, 65), "Input vector has an unexpected shape!"


    # Pass the input vector to the MLP
    # output_vector = model.predict(input_vector_reshaped, verbose=0)[0]  # Get the raw predictions
    output_vector = board.predict_with_mlp(input_vector_reshaped, model)


    # Get the mask for valid moves
    mask = board.get_mask(board.current_turn)

    # Apply the mask to the MLP output
    masked_output = np.multiply(output_vector, mask)

    # Normalize the remaining values (divide by the sum of the masked outputs)
    if np.sum(masked_output) > 0:
        masked_output /= np.sum(masked_output)

    # Find the index of the highest value
    best_move_index = np.argmax(masked_output)

    # Translate the index into (row, col) coordinates
    row, col = divmod(best_move_index, 8)

    return row, col
