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
        adjusted_pos = [6.64493202, 6.14729207, 5.89636059, 5.6629769, 5.697591, 4.04090599,
                        5.20219405, 4.81643487, 4.56128288, 4.54062104, 3.0835603, 2.67863286,
                        2.30383731, 3.75761767, 3.80937173, 2.09748239, 3.28702273, 3.00914085,
                        3.98916236, 3.69134038, 3.34849883, 3.01150153, 2.65381396, 1.5778695,
                        1.18101462, 2.29534458, 2.03905422, 1.57561589, 1.18092808, 1.04535096,
                        0.0]
        own_remaining = sum([adjusted_pos[pos]for pos in my_positions])
        
        # Feature 4: Minimum distance to opponent's mals
        feature4 = 0
        active_enemy_positions = [pos for pos in enemy_positions if pos != yut.rule.FINISHED]
        outcomes, probs = yut.rule.enumerate_all_cast_outcomes(depth=1)
        if active_enemy_positions:
            for enemy_pos in active_enemy_positions:
                for outcome, prob in zip( outcomes, probs ):
                    outcome = outcome[0]
                    pos_true = yut.rule.next_position( enemy_pos, outcome, True )
                    pos_false = yut.rule.next_position( enemy_pos, outcome, False )
                    if pos_true == new_pos:
                        feature4 += prob
                    if pos_false == new_pos and pos_true != pos_false:
                        feature4 += prob
        # Feature 5: Sum of remaining steps for all opponent's mals
        opponent_remaining = sum([adjusted_pos[pos] for pos in enemy_positions])
        
        logging.debug(f"Extracted Features: {feature1}, {feature2}, {own_remaining}, {feature4}, {opponent_remaining}")
        
        return [feature1, feature2, own_remaining, feature4, opponent_remaining]

def generate_training_data(num_games=1000, seed=42, max_ratio=1.0):
    """
    Generates training data by simulating games and balancing the dataset.

    Parameters:
    - num_games (int): Number of games to simulate.
    - seed (int): Random seed for reproducibility.
    - max_ratio (float): Maximum ratio of losing samples to winning samples (e.g., 1.0 for equal numbers).

    Returns:
    - tuple: (features_balanced, labels_balanced)
    """
    event_listener = DataCollectionListener()
    player1 = ExamplePlayer()  # Your AI
    player2 = ExamplePlayer()  # Baseline opponent
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
            labels.append(1 if player_id == winner else 0)

    logging.info(f"Total actions collected: {len(features)}")
    
    # Separate the data into winning and losing actions
    features_wins = [f for f, l in zip(features, labels) if l == 1]
    labels_wins = [1] * len(features_wins)
    
    features_losses = [f for f, l in zip(features, labels) if l == 0]
    labels_losses = [0] * len(features_losses)
    
    logging.info(f"Number of winning actions: {len(features_wins)}")
    logging.info(f"Number of losing actions before balancing: {len(features_losses)}")
    
    # Determine the number of losing samples to keep based on the desired ratio
    desired_num_losses = int(len(features_wins) * max_ratio)
    logging.info(f"Desired number of losing actions after balancing: {desired_num_losses}")
    
    if len(features_losses) > desired_num_losses:
        # Randomly select a subset of losing actions
        np.random.seed(seed)  # For reproducibility
        indices = np.random.choice(len(features_losses), desired_num_losses, replace=False)
        features_losses_balanced = [features_losses[i] for i in indices]
        labels_losses_balanced = [0] * desired_num_losses
        logging.info(f"Reduced losing actions from {len(features_losses)} to {len(features_losses_balanced)}")
    else:
        # If already balanced or losses are fewer than desired, keep all
        features_losses_balanced = features_losses
        labels_losses_balanced = labels_losses
        logging.info(f"No reduction needed for losing actions.")
    
    # Combine the balanced winning and losing actions
    features_balanced = features_wins + features_losses_balanced
    labels_balanced = labels_wins + labels_losses_balanced
    
    logging.info(f"Balanced dataset size: {len(features_balanced)}")
    
    return features_balanced, labels_balanced

if __name__ == "__main__":
    features, labels = generate_training_data(num_games=1000, seed=42, max_ratio=1.0)  # max_ratio=1.0 for equal classes
    # Save the data for training
    with open('training_data_balanced.pkl', 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)
    logging.info("Balanced training data saved to 'training_data_balanced.pkl'.")