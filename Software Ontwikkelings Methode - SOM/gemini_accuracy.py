import os
import pandas as pd
import time
from Lab01 import BoardGameMechanicsAnalyzer

# define variables
GeminiKey = "AIzaSyBFN_04KlBFX8UtnCSWOFZm_98Tm-Q_x00"
BGG = "bgg_dataset.csv"

class RandomGameAnalyzer(BoardGameMechanicsAnalyzer):
    def __init__(self, key, data, output_file="mechanics_scores.csv", num_games=10):
        super().__init__(key, data)  # Initialize the parent class
        self.num_games = num_games  # Number of games to sample
        self.output_file = output_file  # File to track mechanics scores
        self.mechanics_scores = self._load_or_initialize_scores()  # Load or initialize scores

    def _load_or_initialize_scores(self):
        
        # Load existing mechanics scores from a CSV file or initialize a new dictionary.
        
        if os.path.exists(self.output_file):
            mechanics_df = pd.read_csv(self.output_file)
            return dict(zip(mechanics_df["Mechanic"], mechanics_df["Score"]))
        else:
            return {}  # Initialize an empty dictionary if file doesn't exist

    def _update_mechanics_scores(self, dataset_mechanics, gemini_mechanics):
        
        # Update mechanics scores based on their presence in Gemini's predictions.
        
        for mechanic in dataset_mechanics:
            if mechanic in gemini_mechanics:
                self.mechanics_scores[mechanic] = self.mechanics_scores.get(mechanic, 0) + 1
            else:
                self.mechanics_scores[mechanic] = self.mechanics_scores.get(mechanic, 0) - 1

        for mechanic in gemini_mechanics:
            if mechanic not in self.mechanics_scores:
                self.mechanics_scores[mechanic] = 1  # Initialize new mechanics with a score of 1

    def export_mechanics_scores(self):
        
        # Export the mechanics scores to a CSV file.
        
        mechanics_df = pd.DataFrame(
            list(self.mechanics_scores.items()), columns=["Mechanic", "Score"]
        )
        mechanics_df.to_csv(self.output_file, index=False)
        print(f"Mechanics scores saved to {self.output_file}")