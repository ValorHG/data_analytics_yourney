import time
from gemini_accuracy import RandomGameAnalyzer

#delete mechanics_scores to start clean

# define variables
GeminiKey = "AIzaSyBFN_04KlBFX8UtnCSWOFZm_98Tm-Q_x00"
BGG = "bgg_dataset.csv"

class TopRatedGameAnalyzer(RandomGameAnalyzer):
    def __init__(self, key, data, output_file="mechanics_scores.csv", num_games_per_batch=10):
        super().__init__(key, data, output_file, num_games=num_games_per_batch)
        self.accuracies = []  # willstore accuracy values for each game

    def preprocess_top_rated_games(self, top_n=200):
        """
        Preprocess the dataset to select the top N highest-rated games.
        """
        # convert `Rating Average` to numeric
        self.Data["Rating Average"] = self.Data["Rating Average"].str.replace(",", ".").astype(float)
        # select top N games
        self.top_rated_games = self.Data.nlargest(top_n, "Rating Average")

    def process_top_rated_games(self, duration_minutes=20):
        """
        Process the top-rated games in batches of 10 per minute and calculate the mean accuracy.
        """
        start_time = time.time()
        duration = duration_minutes * 60  # convert duration to seconds
        total_games = len(self.top_rated_games)
        games_processed = 0

        while time.time() - start_time < duration and games_processed < total_games:

            
            # get the next batch of games
            batch = self.top_rated_games.iloc[games_processed:games_processed + self.num_games]
            for index, row in batch.iterrows():

                

                game_name = row["Name"]
                dataset_mechanics = [m.strip() for m in row["Mechanics"].split(",")]

                print(f"\nAnalyzing game: {game_name}")

                # send the prompt to Gemini
                gemini_input = (
                    f'The game "{game_name}" has the following mechanics according to my database: "{row["Mechanics"]}". '
                    f'Give me a python list that has the mechanics that guaranteed appear in the game, from the mechanics given. If you have to remove some mechanics do that, dont be afraid. I want to know of the mechnics listed, are actually in the game according to you. '
                    f'Do not include explanations or extra text, only the Python list.'
                )
                response = self.chat_with_gemini(gemini_input)
                gemini_mechanics = self.extract_python_list(response)

                if not gemini_mechanics:
                    print(f"Failed to extract mechanics for game: {game_name}")
                    continue

                # calculate accuracy
                total_mechanics = len(dataset_mechanics)
                total_predictions = len(gemini_mechanics)
                accuracy = total_predictions / total_mechanics if total_mechanics > 0 else 0
                self.accuracies.append(accuracy)

                print("Total mechanics (from dataset):", total_mechanics)
                print("Gemini's predicted mechanics (count):", total_predictions)
                print(f"Accuracy for {game_name}: {accuracy:.2%}")

                # update mechanics scores
                self._update_mechanics_scores(dataset_mechanics, gemini_mechanics)

            
            # change processed games and export progress
            games_processed += self.num_games
            self.export_mechanics_scores()

            # wait for the next batch
            if games_processed < total_games:
                elapsed_time = time.time() - start_time
                if elapsed_time % 60 < 50:
                    time.sleep(70 - (elapsed_time % 60))  # lleep until the next 70 seconds

        # calculate mean accuracy
        mean_accuracy = sum(self.accuracies) / len(self.accuracies) if self.accuracies else 0
        print(f"\nMean Accuracy across all games: {mean_accuracy:.2%}")
        return mean_accuracy
    
# create an instance of the new analyzer
top_rated_analyzer = TopRatedGameAnalyzer(GeminiKey, BGG, output_file="mechanics_scores.csv")

# preprocess the dataset to get the top 200 games
top_rated_analyzer.preprocess_top_rated_games(top_n=200)

# process these games and calculate the mean accuracy
mean_accuracy = top_rated_analyzer.process_top_rated_games(duration_minutes=20)
print(f"Mean Accuracy: {mean_accuracy:.2%}")
