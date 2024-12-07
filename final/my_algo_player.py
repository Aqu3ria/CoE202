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
            self.model.load_state_dict(torch.load('yut_score_model.pth', map_location=torch.device('cpu')))
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
        if active_enemy_positions:
            distances = [abs(new_pos - enemy_pos) for enemy_pos in active_enemy_positions]
            feature4 = min(distances)
        else:
            feature4 = 30  # Max distance if no active enemies
        
        # Feature 5: Sum of remaining steps for all opponent's mals
        opponent_remaining = sum([30 - pos if pos < 30 else 0 for pos in enemy_positions])
        
        logging.debug(f"Extracted Features: {feature1}, {feature2}, {own_remaining}, {feature4}, {opponent_remaining}")
        
        return [feature1, feature2, own_remaining, feature4, opponent_remaining]

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