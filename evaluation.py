# evaluation.py
def naive_evaluation(board):
    white_score = sum(cell == 1 for row in board.board for cell in row)
    black_score = sum(cell == -1 for row in board.board for cell in row)
    return white_score - black_score


def weighted_positional_value(board):
    # Weighted evaluation considering strategic positions on the board.
    # Define a positional weight matrix for Othello.
    weights = [
        [100, -20, 10, 5, 5, 10, -20, 100],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [10, -2, 5, 1, 1, 5, -2, 10],
        [5, -2, 1, 0, 0, 1, -2, 5],
        [5, -2, 1, 0, 0, 1, -2, 5],
        [10, -2, 5, 1, 1, 5, -2, 10],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [100, -20, 10, 5, 5, 10, -20, 100]
    ]
    
    white_score = 0
    black_score = 0
    
    for row in range(8):
        for col in range(8):
            if board.board[row][col] == 1:
                white_score += weights[row][col]
            elif board.board[row][col] == -1:
                black_score += weights[row][col]
    
    return white_score - black_score

def stability_heuristic(board):
    # Stability heuristic to evaluate stable pieces that cannot be flipped.
    # Corners are the most stable, followed by edge pieces if they are fully anchored.
    stability_matrix = [
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1]
    ]
    
    white_stability = 0
    black_stability = 0
    
    for row in range(8):
        for col in range(8):
            if board.board[row][col] == 1:
                white_stability += stability_matrix[row][col]
            elif board.board[row][col] == -1:
                black_stability += stability_matrix[row][col]
    
    return white_stability - black_stability

def mobility_heuristic(board, color):
    # Mobility heuristic to evaluate the number of valid moves available.
    opponent_color = -color
    player_mobility = len(board.get_valid_moves(color))
    opponent_mobility = len(board.get_valid_moves(opponent_color))
    return player_mobility - opponent_mobility


def frontier_heuristic(board):
    # Frontier heuristic to evaluate the number of frontier discs (pieces adjacent to empty spaces).
    directions = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]
    white_frontier = 0
    black_frontier = 0

    for row in range(8):
        for col in range(8):
            if board.board[row][col] != 0:
                for dx, dy in directions:
                    x, y = row + dx, col + dy
                    if 0 <= x < 8 and 0 <= y < 8 and board.board[x][y] == 0:
                        if board.board[row][col] == 1:
                            white_frontier += 1
                        elif board.board[row][col] == -1:
                            black_frontier += 1
                        break

    return black_frontier - white_frontier
    
def evaluate_all(board):
    heuristic_names = ['naive', 'weighted', 'stability', 'mobility', 'frontier']
    total_score = 0
    for heuristic in heuristic_names:
        total_score += evaluate_board(board, heuristic)
    return total_score

# Handler for different evaluation heuristics.
def evaluate_board(board, heuristic_type):
    if heuristic_type == 'naive':
        return naive_evaluation(board)
    elif heuristic_type == 'weighted':
        return weighted_positional_value(board)
    elif heuristic_type == 'stability':
        return stability_heuristic(board)
    elif heuristic_type == 'mobility':
        return mobility_heuristic(board, board.current_turn)
    elif heuristic_type == 'frontier':
        return frontier_heuristic(board)
    elif heuristic_type == 'all':
        return evaluate_all(board)
