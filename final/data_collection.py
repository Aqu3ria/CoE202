import yut.engine
import torch
import numpy as np
import pickle
from example_player import ExamplePlayer  # Ensure you have this implemented
from my_algo_player import MyAlgo  # Placeholder for your AI class
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DataCollectionListener:
    def __init__(self):
        self.game_history = []  # List to store all games' data
        self.current_game = {'player_actions': [], 'result': None}
        self.current_player = None  # To track which player is making the move

    def on_game_start(self, name1, name2):
        self.current_game = {'player_actions': [], 'result': None}
        self.player_names = (name1, name2)
        logging.info(f"Game started between {name1} and {name2}")

    def on_turn_begin(self, turn):
        # Determine current player based on turn
        # Assuming player1 starts first (turn=0)
        self.current_player = 0 if turn % 2 == 0 else 1
        logging.debug(f"Turn {turn} begins. Current player: Player {self.current_player + 1}")

    def on_yut_cast(self, cast_outcome):
        pass  # Not used for data collection

    def on_state(self, state):
        pass  # Not used for data collection

    def on_action(self, action, result):
        mal_to_move, yutscore_to_use, shortcut, debug_msg = action
        legal_move, my_positions, enemy_positions, num_mals_caught = result

        logging.debug(f"Player {self.current_player + 1} Action: Move Mal {mal_to_move}, Yutscore {yutscore_to_use}, Shortcut {shortcut}")
        logging.debug(f"New Positions - My Mals: {my_positions}, Enemy Mals: {enemy_positions}")

        # Extract features based on the action and state
        features = self.extract_features(mal_to_move, yutscore_to_use, shortcut, my_positions, enemy_positions)

        # Append action data with current player
        self.current_game['player_actions'].append({
            'player': self.current_player,
            'features': features
        })

    def on_game_end(self, winner):
        self.current_game['result'] = winner
        self.game_history.append(self.current_game)
        logging.info(f"Game ended. Winner: Player {winner + 1}")

    def extract_features(self, mal_to_move, yutscore, shortcut, my_positions, enemy_positions):
        """
        Extracts features based on the current game state and the chosen action.

        Parameters:
        - mal_to_move (int): Index of the mal to move.
        - yutscore (int): Yut score used for the move.
        - shortcut (bool): Whether a shortcut is used.
        - my_positions (tuple): Positions of the AI's mals.
        - enemy_positions (tuple): Positions of the opponent's mals.

        Returns:
        - list: Feature vector.
        """
        # Feature 1: Can use a shortcut (1 or 0)
        feature1 = int(shortcut)
        
        # Feature 2: Can capture an opponent's mal (1 or 0)
        new_pos = yut.rule.next_position(my_positions[mal_to_move], yutscore, shortcut)
        feature2 = int(new_pos in list(enemy_positions))
        
        # Feature 3: Sum of remaining steps for all own mals
        own_remaining = sum([30 - pos if pos < 30 else 0 for pos in my_positions])
        
        # Feature 4: Minimum distance to opponent's mals
        active_enemy_positions = [pos for pos in enemy_positions if pos != yut.rule.FINISHED]
        if active_enemy_positions:
            distances = [abs(new_pos - enemy_pos) for enemy_pos in active_enemy_positions]
            feature4 = min(distances)
        else:
            feature4 = 30  # Max distance if no active enemies
        
        # Feature 5: Sum of remaining steps for all opponent's mals
        opponent_remaining = sum([30 - pos if pos < 30 else 0 for pos in enemy_positions])
        
        logging.debug(f"Extracted Features: {feature1}, {feature2}, {own_remaining}, {feature4}, {opponent_remaining}")
        
        return [feature1, feature2, own_remaining, feature4, opponent_remaining]

def generate_training_data(num_games=1000, seed=42):
    event_listener = DataCollectionListener()
    player1 = MyAlgo()  # Your AI
    player2 = ExamplePlayer() # Baseline opponent
    engine = yut.engine.GameEngine()

    for game_num in range(num_games):
        engine.play(player1, player2, seed=seed + game_num, game_event_listener=event_listener)
        if (game_num + 1) % 100 == 0:
            logging.info(f"Completed {game_num + 1}/{num_games} games.")

    # Prepare features and labels
    features = []
    labels = []

    for game in event_listener.game_history:
        winner = game['result']
        for action_data in game['player_actions']:
            player_id = action_data['player']
            features.append(action_data['features'])
            # Label is 1 if the player who took the action won, else 0
            labels.append(1 if player_id == winner else 0)

    logging.info(f"Generated {len(features)} feature-action pairs.")
    return features, labels

if __name__ == "__main__":
    features, labels = generate_training_data(num_games=1000)
    # Save the data for training
    with open('training_data.pkl', 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)
    logging.info("Training data saved to 'training_data.pkl'.")