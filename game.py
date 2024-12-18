# game.py
from board import OthelloBoard

def play():
    # Print game mode options
    print("Choose a game mode:")
    print("1. 2 Player")
    print("2. 1 Player vs Computer")
    print("3. Computer vs Computer")
    game_mode = input("Enter option (1/2/3): ")
    
    agent_options = {
        "1": "naive",
        "2": "minimax",
        "3": "alpha_beta_minimax",
        "4": "expectiminimax",
        "5": "monte_carlo",
        "6": "MLP"
    }


    heuristic_options = {
        "1": "naive",
        "2": "weighted",
        "3": "stability",
        "4": "mobility",
        "5": "frontier",
        "6": "all"
    }

    agent_w = None
    agent_b = None
    heuristic_w = 'naive'
    heuristic_b = 'naive'

    if game_mode == '2':
        print("Choose computer agent:")
        for key, value in agent_options.items():
            print(f"{key}. {value}")
        agent_b = agent_options.get(input("Enter option (1/2/3/4/5/6): "), "naive")
        
        print("Choose heuristic for computer agent:")
        for key, value in heuristic_options.items():
            print(f"{key}. {value}")
        heuristic_b = heuristic_options.get(input("Enter option (1/2/3/4/5/6): "), "naive")

    elif game_mode == '3':
        print("Choose computer agent for Black player:")
        for key, value in agent_options.items():
            print(f"{key}. {value}")
        agent_b = agent_options.get(input("Enter option (1/2/3/4/5/6): "), "naive")

        print("Choose heuristic for Black player:")
        for key, value in heuristic_options.items():
            print(f"{key}. {value}")
        heuristic_b = heuristic_options.get(input("Enter option (1/2/3/4/5/6): "), "naive")
        
        print("Choose computer agent for White player:")
        for key, value in agent_options.items():
            print(f"{key}. {value}")
        agent_w = agent_options.get(input("Enter option (1/2/3/4/5/6): "), "naive")

        print("Choose heuristic for White player:")
        for key, value in heuristic_options.items():
            print(f"{key}. {value}")
        heuristic_w = heuristic_options.get(input("Enter option (1/2/3/4/5/6): "), "naive")


    board = OthelloBoard()
    board.start_game(game_mode, agent_w, agent_b, heuristic_w, heuristic_b)


# if __name__ == "__main__":
#     play()


if __name__ == "__main__":
    total_games = 100
    mcts_wins = 0
    mlp_wins = 0
    draws = 0

    results = {
        "mlp_first": {"mlp_wins": 0, "mcts_wins": 0, "draws": 0},
        "mcts_first": {"mlp_wins": 0, "mcts_wins": 0, "draws": 0},
    }

    for i in range(total_games):
        print(f"Playing game {i + 1}/{total_games}...")

        if i % 2 == 0:  # MLP goes first (Black)
            agent_b = "MLP"
            agent_w = "monte_carlo"
            first_player = "mlp_first"
        else:  # MCTS goes first (Black)
            agent_b = "monte_carlo"
            agent_w = "MLP"
            first_player = "mcts_first"

        board = OthelloBoard()
        winner = board.start_game(game_mode="3", agent_w=agent_w, agent_b=agent_b)

        if winner == 1:  # Black (agent_b) wins
            if agent_b == "MLP":
                mlp_wins += 1
                results[first_player]["mlp_wins"] += 1
            else:
                mcts_wins += 1
                results[first_player]["mcts_wins"] += 1
        elif winner == -1:  # White (agent_w) wins
            if agent_w == "MLP":
                mlp_wins += 1
                results[first_player]["mlp_wins"] += 1
            else:
                mcts_wins += 1
                results[first_player]["mcts_wins"] += 1
        else:  # Draw
            draws += 1
            results[first_player]["draws"] += 1

    # Print final results
    print("Final Results:")
    print(f"MCTS Total Wins: {mcts_wins}")
    print(f"MLP Total Wins: {mlp_wins}")
    print(f"Draws: {draws}")
    print("\nDetailed Results:")
    print("MLP First:")
    print(f"  MLP Wins: {results['mlp_first']['mlp_wins']}")
    print(f"  MCTS Wins: {results['mlp_first']['mcts_wins']}")
    print(f"  Draws: {results['mlp_first']['draws']}")
    print("MCTS First:")
    print(f"  MLP Wins: {results['mcts_first']['mlp_wins']}")
    print(f"  MCTS Wins: {results['mcts_first']['mcts_wins']}")
    print(f"  Draws: {results['mcts_first']['draws']}")
