import yut.engine
import torch
import torch.nn.functional as F
import numpy as np
from model_definition import YutScoreModel  # Ensure this matches your model definition
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MyAlgo(yut.engine.Player):
    def __init__(self):
        super().__init__()
        self.model = YutScoreModel()
        try:
            self.model.load_state_dict(torch.load('yut_score_model_balanced.pth', map_location=torch.device('cpu')))
            self.model.eval()
            logging.info("Model loaded successfully.")
        except FileNotFoundError:
            logging.error("Model file 'yut_score_model.pth' not found. Please train the model first.")
            self.model = None

    def name(self):
        return "ScoreBasedDeepLearningAI"

    def reset(self, random_state):
        self.random_state = random_state

    def action(self, state):

        turn, my_positions, enemy_positions, available_yutscores = state
        possible_actions = self.generate_possible_actions(my_positions, enemy_positions, available_yutscores)

        if self.model is None:

            return possible_actions[0][0], possible_actions[0][1], possible_actions[0][2], ""
        

        if not possible_actions:
            return possible_actions[0][0], possible_actions[0][1], possible_actions[0][2], ""
        
        best_score = -float('inf')
        best_action = None
        debug_msg = ""
        
        for action in possible_actions:
            mal_to_move, yutscore_to_use, shortcut = action
            features = self.extract_features(mal_to_move, yutscore_to_use, shortcut, my_positions, enemy_positions)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Batch size 0
            with torch.no_grad():
                output = self.model(features_tensor)
                score = torch.sigmoid(output).item()  # Convert to probability
            if score > best_score:
                best_score = score
                best_action = action
                debug_msg = f"Chosen action with score {score:.2f}"

        
        
        if best_action:
            logging.debug(f"Selected action: Mal {best_action[0]}, Yutscore {best_action[1]}, Shortcut {best_action[2]}")
            return (*best_action, debug_msg)
        else:
            logging.info("No legal actions available.")
            return 

    def generate_possible_actions(self, my_positions, enemy_positions, available_yutscores):
        possible_actions = []
        for mi, mp in enumerate(my_positions):
            if mp == yut.rule.FINISHED:
                continue
            for ys in available_yutscores:
                for shortcut in [True, False]:
                    legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move( my_positions, enemy_positions, mi, ys, shortcut )
                    if legal_move:
                        possible_actions.append((mi, ys, shortcut))

        return possible_actions

    def should_use_shortcut(self, position):
        # Positions where shortcuts are mandatory
        mandatory_shortcuts = {5: 13, 10: 11, 15: 23}
        return position in mandatory_shortcuts


    def random_fallback_action(self, state):
        turn, my_positions, enemy_positions, available_yutscores = state
        yutscore_to_use = np.random.choice(available_yutscores)
        available_mals = [i for i, pos in enumerate(my_positions) if pos != yut.rule.FINISHED]
        mal_to_move = np.random.choice(available_mals)
        shortcut = True
        debug_msg = "Random fallback action"
        logging.debug(f"Random fallback action: Mal {mal_to_move}, Yutscore {yutscore_to_use}, Shortcut {shortcut}")
        return mal_to_move, yutscore_to_use, shortcut, debug_msg

    def handle_no_legal_actions(self, state):
        # Implement logic to handle no available legal actions
        # For example, return a default legal action or handle as a loss
        # Here, we'll return a 'backdo' knowing it's illegal to trigger loss
        logging.warning("No legal actions available. Attempting an illegal 'backdo' to trigger loss.")
        return (0, -1, False, "Attempting illegal 'backdo' to trigger loss")


    def on_my_action(self, state, my_action, result):
        pass

    def on_enemy_action(self, state, enemy_action, result):
        pass

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


    '''
   def extract_features(self, mal_to_move, yutscore, shortcut, my_positions, enemy_positions):
        # Feature 1: Can use a shortcut (1 or 0)
        feature1 = int(shortcut)
        
        # Feature 2: Can capture an opponent's mal (1 or 0)
        new_pos = yut.rule.next_position(my_positions[mal_to_move], yutscore, shortcut)
        feature2 = int(new_pos in list(enemy_positions))
        
        # Feature 3: Sum of remaining steps for all own mals
        own_remaining = sum([30 - pos if pos < 30 else 0 for pos in my_positions])
        
        # Feature 4: Minimum distance to opponent's mals
        active_enemy_positions = [pos for pos in enemy_positions if pos != yut.rule.FINISHED]
        if active_enemy_positions :
            distances = [abs(new_pos - enemy_pos) for enemy_pos in active_enemy_positions]
            feature4 = min(distances)
        else:
            feature4 = 30  # Max distance if no active enemies
        
        # Feature 5: Sum of remaining steps for all opponent's mals
        opponent_remaining = sum([30 - pos if pos < 30 else 0 for pos in enemy_positions])
        
        logging.debug(f"Extracted Features: {feature1}, {feature2}, {own_remaining}, {feature4}, {opponent_remaining}")
        
        return [feature1, feature2, own_remaining, feature4, opponent_remaining]

    '''