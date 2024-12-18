import time
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import load_model
from agents import naive, minimax, alpha_beta_minimax, expectiminimax, monte_carlo, mlp


class OthelloBoard:
    def __init__(self, c=1.41):
        # Initialize an 8x8 board
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        
        # Set the starting positions
        # 1 represents black, -1 represents white, 0 represents an empty cell
        self.board[3][3] = -1   # White
        self.board[3][4] = 1  # Black
        self.board[4][3] = 1  # Black
        self.board[4][4] = -1   # White
        
        # Start with black's turn (1)
        self.current_turn = 1
        
        # Track piece counts for convenience
        self.piece_count = {1: 2, -1: 2}

        self.c = c

        # if not hasattr(self, 'model'):  # Avoid reloading the model
        #     self.model = load_model('othello_mlp_trained.h5')
        
    def display_board(self):
        # Utility to visually represent the board (useful for debugging)
        print("  " + " ".join(map(str, range(8))))  # Print column indices
        for i, row in enumerate(self.board):
            row_str = " ".join(['B' if cell == 1 else 'W' if cell == -1 else '.' for cell in row])
            print(f"{i} {row_str}")  # Print row index followed by the row content
        print()
    
    def copy(self):
        new_board = OthelloBoard()
        new_board.board = [row[:] for row in self.board]
        new_board.current_turn = self.current_turn
        new_board.piece_count = self.piece_count.copy()
        return new_board

    def is_valid_move(self, row, col, color):
        # Directions for 8 possible neighbors (N, NE, E, SE, S, SW, W, NW)
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        # Check if the cell is empty
        if self.board[row][col] != 0:
            return False
        
        # Check in all directions for valid captures
        opponent_color = -color
        for dx, dy in directions:
            x, y = row + dx, col + dy
            found_opponent = False
            
            # Move in the direction while it's within bounds
            while 0 <= x < 8 and 0 <= y < 8:
                if self.board[x][y] == opponent_color:
                    found_opponent = True
                    x += dx
                    y += dy
                elif self.board[x][y] == color and found_opponent:
                    # Found a valid move that sandwiches opponent pieces
                    return True
                else:
                    break
        
        return False

    def get_valid_moves(self, color):
        return [(row, col) for row in range(8) for col in range(8) if self.is_valid_move(row, col, color)]

    def execute_move(self, row, col, color):
        # Place the piece, flip the captures, and return capture count
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        if not self.is_valid_move(row, col, color):
            return False
        
        # Place the piece
        self.board[row][col] = color
        capture_count = 0
        
        # Check in all directions for valid captures
        opponent_color = -color
        for dx, dy in directions:
            x, y = row + dx, col + dy
            opponent_cells = []
            
            # Move in the direction while it's within bounds
            while 0 <= x < 8 and 0 <= y < 8:
                if self.board[x][y] == opponent_color:
                    opponent_cells.append((x, y))
                    x += dx
                    y += dy
                elif self.board[x][y] == color:
                    for i, j in opponent_cells:
                        self.board[i][j] = color
                        capture_count += 1
                    break
                else:
                    break
        
        # Update piece counts
        self.piece_count[color] += capture_count + 1
        self.piece_count[opponent_color] -= capture_count
        
        return True

    def get_move_score(self, row, col, color):
        if not self.is_valid_move(row, col, color):
            return 0
        
        score = 1  # Initial score for placing the piece itself
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        opponent_color = -color
        
        for dx, dy in directions:
            x, y = row + dx, col + dy
            opponent_cells = []
            
            while 0 <= x < 8 and 0 <= y < 8:
                if self.board[x][y] == opponent_color:
                    opponent_cells.append((x, y))
                    x += dx
                    y += dy
                elif self.board[x][y] == color:
                    score += len(opponent_cells)
                    break
                else:
                    break
        
        return score

    def alternate_turns(self):
        # Alternate player turns, skip if no valid moves
        self.current_turn = -self.current_turn
        if not self.get_valid_moves(self.current_turn):
            self.current_turn = -self.current_turn

    def is_game_over(self):
        # Check if no moves are left for either player
        return len(self.get_valid_moves(1)) == 0 and len(self.get_valid_moves(-1)) == 0

    def get_winner(self):
        """
        Count pieces on the board and determine the winner.
        :return: (1, count) if Black wins, (-1, count) if White wins, (0, count) for a draw.
        """
        black = sum(cell == 1 for row in self.board for cell in row)  # Black is 1
        white = sum(cell == -1 for row in self.board for cell in row)  # White is -1

        if black > white:
            return 1, black  # Black wins
        elif white > black:
            return -1, white  # White wins
        else:
            return 0, black  # Draw, return either count (they are equal)

    def vectorize_board(self):
        """
        Return the board as a 1D vector suitable for the MLP input.
        The vector includes:
        - Flattened board state (row-major order).
        - Current player's turn as the last element.
        
        Returns:
            np.ndarray: 1D array of shape (65,).
        """
        # Flatten the board into a 1D array
        board_state = np.array([cell for row in self.board for cell in row])
        
        # Append the current turn
        player_turn = np.array([self.current_turn])
        
        # Combine into a single input vector
        input_vector = np.append(board_state, player_turn)
        
        return input_vector


    def get_mask(self, color):
        # Create a vectorized board listing possible moves for masking MLP output
        mask = [0] * 64
        valid_moves = self.get_valid_moves(color)
        for row, col in valid_moves:
            mask[row * 8 + col] = 1
        return mask
    

    # @tf.function(reduce_retracing=True)
    def predict_with_mlp(self, input_vector, model):
        input_vector_tensor = tf.convert_to_tensor(input_vector, dtype=tf.float32)
        # print(f"Input tensor shape: {input_vector_tensor.shape}")
        output_vector = model.predict(input_vector_tensor, verbose=0)[0]
        # print(f"Output vector shape: {output_vector.shape}")

        return output_vector



    def get_next_move(self, color, agent, heuristic='naive', depth=6, model=None):
        # Selects next move based on a given strategy
        if agent == 'naive':
            return naive(self, color, heuristic)
        elif agent == 'minimax':
            _, move = minimax(self, depth, color, heuristic)
            return move[0], move[1]
        elif agent == 'alpha_beta_minimax':
            _, move = alpha_beta_minimax(self, depth, float('-inf'), float('inf'), color, heuristic)
            return move[0], move[1]
        elif agent == 'expectiminimax':
            _, move = expectiminimax(self, depth, color, heuristic)
            return move[0], move[1]
        elif agent == 'monte_carlo':
            return monte_carlo(self, color, 1000, self.c)
        elif agent == 'MLP':
            return mlp(self, model)
        
    
    def start_game(self, game_mode='2 Player', agent_w=None, agent_b=None, heuristic_w='naive', heuristic_b='naive', display=True):
        # Start the game based on the chosen game mode
        if display:
            self.display_board()
            time.sleep(0.1)

        model = load_model('othello_mlp_128_256_512_1024_2048_2048_1024_512_256_128_64_overfit.h5')
        
        while not self.is_game_over():
            if not any(self.get_mask(self.current_turn)):
                print(f"No valid moves for {'Black' if self.current_turn == 1 else 'White'}, skipping turn.")
                continue

            if game_mode == '1':
                # Human player input
                move = input(f"Player {'B' if self.current_turn == 1 else 'W'} enter your move (row col): ")
                try:
                    row, col = map(int, move.split())
                    if not self.is_valid_move(row, col, self.current_turn):
                        print("Invalid move. Try again.")
                        continue
                except ValueError:
                    print("Invalid input format. Please enter row and column numbers separated by a space.")
                    continue
            elif game_mode == '2':
                if self.current_turn == 1:
                    # Human Turn
                    move = input(f"Player 'B' enter your move (row col): ")
                    try:
                        row, col = map(int, move.split())
                        if not self.is_valid_move(row, col, self.current_turn):
                            print("Invalid move. Try again.")
                            continue
                    except ValueError:
                        print("Invalid input format. Please enter row and column numbers separated by a space.")
                        continue
                else:
                    # Computer Turn
                    print("Computer's Turn (White)")
                    row, col = self.get_next_move(self.current_turn, agent_b, heuristic_b)
                    time.sleep(0.5)
                    print(f'Computer to play {row, col}')
            elif game_mode == '3':
                # Computer vs Computer
                if display:
                    print(f"Computer {'B' if self.current_turn == 1 else 'W'}'s Turn")
                if self.current_turn == 1:
                    row, col = self.get_next_move(self.current_turn, agent_b, heuristic_w, 6, model)
                else:
                    row, col = self.get_next_move(self.current_turn, agent_w, heuristic_b, 6, model)
                # time.sleep(0.1)
                if display:
                    print(f'Computer to play {row, col}')
    
            if row == -1 and col == -1:
                print(f"Player {'B' if self.current_turn == 1 else 'W'} has no valid moves, skipping turn.")
                self.alternate_turns()
                continue

            # Execute the move
            self.execute_move(row, col, self.current_turn)
            if display:
                self.display_board()
            self.alternate_turns()
        
        # Game over, display winner
        winner, score = self.get_winner()
        if winner == 1:
            print(f"Game Over! Winner: Black with score {score}")
        elif winner == -1:
            print(f"Game Over! Winner: White with score {score}")
        else:
            print(f"Game Over! It's a draw with score {score}")
        self.display_board

        return winner