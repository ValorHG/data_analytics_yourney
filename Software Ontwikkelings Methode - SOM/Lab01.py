# import libraries
import google.generativeai as genai
import pandas as pd
import re

# define variables
GeminiKey = "AIzaSyBFN_04KlBFX8UtnCSWOFZm_98Tm-Q_x00"
BGG = "bgg_dataset.csv"

class  BoardGameMechanicsAnalyzer:

# initializes function, this cleans the data (BGG-dataset) Preparing gemini.
  def __init__(self, key, data):
    self.Key = key
    self.Data = pd.read_csv(data, delimiter= ';').dropna()
    self.Model = genai.GenerativeModel("gemini-1.5-flash")
    self.mechanics_list = None
    genai.configure(api_key=GeminiKey)

# takes the prompt as an input and returns gemini's response.
  def chat_with_gemini(self, prompt):
    return(self.Model.generate_content(prompt).text) 
  
# a helper function to extract a python list from gemini
  @staticmethod
  def extract_python_list(response):
    match = re.search(r'\[.*\]', response)
    if match:
      try:
        return eval(match.group(0))
      except Exception as ex:
        print("Error not able to convert", ex)
    else:
      print("No valid python list found")
    return None

# the user gives a game from the data set and the funtion returns the accuracy of gemini is on telling the accuracy of those mechanics
  def calculate_accuracy(self):
    user_input = input("Enter a game from the BGG-dataset: ").strip()
    if user_input in self.Data["Name"].values:
      game_genre = (self.Data).loc[(self.Data)["Name"] == user_input, "Mechanics"].iloc[0]
      
      # count total mechanics from the data set
      total_mechanics = len([m.strip() for m in game_genre.split(",")])

      # ask gemini of they think the mechanics fit the game and return a lis
      gemini_input = f'The game "{user_input}" has the following mechanics: "{game_genre}". GIVE ME A PYTHON LIST OF THE MECHANICS THAT YOU THINK ARE IN THE GAME, DO NOT EXPLAIN, ONLY GIVE ME A PYTHON LIST, NO EXPLANATION OR OTHER NONSENCE'
      response = self.chat_with_gemini(gemini_input)
      print("Gemini says:", response)

      # extract the list from gemini's response
      self.mechanics_list = self.extract_python_list(response)

      # count the total items in the given list and calculate gemini's accuracy
      if self.mechanics_list:
        total_predictions = len(self.mechanics_list)
        accuracy = total_predictions / total_mechanics if total_mechanics > 0 else 0

        print("Total mechanics (from dataset):", total_mechanics)
        print("Gemini's predicted mechanics (count):", total_predictions)
        print(f"Accuracy: {accuracy:.2%}")  # display as a percentage
      else:
        print("Failed to extract mechanics list from Gemini's response.")
        
    else:
      print("The game you provided does not occur in the BGG-dataset")

# the user inputs multiple games with their year, this function returns per game gemini's accuracy. Starts with asking the amount of games
  def multiple_games(self):
    games_asked = {}
    i = int(input("How many games do you want to check: "))
    
    # loop to get the games the user wants to check
    while i > 0:
        user_input_gamename = input("Enter the name of the game: ")
        user_input_gameyear = int(input("Enter the year the game was published: "))
        
        # check if the game exists in the dataset
        if ((self.Data["Name"] == user_input_gamename) & (self.Data["Year Published"] == user_input_gameyear)).any():
            i -= 1
            games_asked[user_input_gamename] = user_input_gameyear
        else:
            print("The data you provided does not match the data in the BGG-dataset.")
    
    if games_asked:  # check if the dict is not empty
       for game_name, game_year in games_asked.items():
        print(f"\nProcessing game: {game_name} ({game_year})")
        
        # find the game in the dataset
        game_data = self.Data[(self.Data["Name"] == game_name) & (self.Data["Year Published"] == game_year)]

        # get the mechanics for the game
        if not game_data.empty:
          game_genre = game_data["Mechanics"].iloc[0]
          total_mechanics = len([m.strip() for m in game_genre.split(",")])

          gemini_input = (
          f'The game "{game_name}" has the following mechanics according to my database: "{game_genre}". '
          f'Give me a python list that has the mechanics that guaranteed appear in the game, from the mechanics given. If you have to remove some mechanics do that, dont be afraid. I want to know of the mechnics listed, are actually in the game according to you. '
          f'Do not include explanations or extra text, only the Python list.'
          )
          
          response = self.chat_with_gemini(gemini_input)
          print("Gemini says:", response)
                
          # extract the list with mechanics
          self.mechanics_list = self.extract_python_list(response)

                # calculate the accuracy from gemini.
          if self.mechanics_list:
            total_predictions = len(self.mechanics_list)
            accuracy = total_predictions / total_mechanics if total_mechanics > 0 else 0

            print("Total mechanics (from dataset):", total_mechanics)
            print("Gemini's predicted mechanics (count):", total_predictions)
            print(f"Accuracy for {game_name}: {accuracy:.2%}")
          else:
            print(f"Failed to extract mechanics list from Gemini's response for {game_name}.")
    else:
      print(f"Could not find {game_name} ({game_year}) in the dataset.")
      
# prepare class
#p1 = BoardGameMechanicsAnalyzer(GeminiKey, BGG)

# execute functions
#p1.calculate_accuracy()
#p1.multiple_games()