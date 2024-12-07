import yut.engine
from my_algo_player import MyAlgo
from example_player import ExamplePlayer

def simulate_games(num_games=500):
    player1 = MyAlgo()
    player2 = ExamplePlayer()
    engine = yut.engine.GameEngine()
    
    win_count = 0
    for game_num in range(num_games):
        winner = engine.play(player1, player2)
        if winner == 0:
            win_count += 1
    
    win_rate = (win_count / num_games) * 100
    print(f"MyAlgo win rate: {win_rate:.2f}% over {num_games} games.")

if __name__ == "__main__":
    simulate_games(num_games=500)